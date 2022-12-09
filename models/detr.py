# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


# bbox最后的输出头
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 最后一层的后面不需要激活函数，其他的有激活函数
        # 因为最后一层可能会作为直接的输出层，那么在最后的输出变成目标值之前的激活函数可能并不需要ReLU这种
        # 也可能是简单的一个Sigmoid就可以了，所以最后不需要激活函数
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            一张图片中最大的检测数量
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        # transformer内特征的维度
        hidden_dim = transformer.d_model
        # 最后的分类头
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 最后的bbox的头
        # MLP的输出output_dim 是4，是bbox的四个值
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 1x1 衔接 backbone和transformer
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # backbone中的Joiner
        self.backbone = backbone
        # 辅助Loss
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 1. 先经过backbone
        # backbone中的Joiner，第二个返回值就是pos
        # pos 是给transformer的 位置嵌入
        # features是一个list，每一个item是不同的backbone的layer
        # 每个item是NestedTesnor, NestedTensor 包含mask和tensors
        features, pos = self.backbone(samples)
        # 2. 分解出feature和mask src [bs,2048,h,w]
        # src like [bs,2048,h,w]
        src, mask = features[-1].decompose()

        assert mask is not None
        # 3. 传入transformer

        # input_proj [bs,2048,h,w] --input_proj--> [bs,256,h,w]
        # query_embed.weight [100,256]
        # pos只使用最后一个位置编码, src可能会有多层（对于全景分割任务，pos与之对应有多个）
        # [0] hs返回两个返回值，0就是transformer decoder的输出，1是encoder的输出 hs (6,bs,100,256)
        # 这里第二个返回值是encoder的输出，没有使用，seg使用了第二个返回值
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # outpus_class (6,bs,100,92)
        # 类别的输出
        outputs_class = self.class_embed(hs)
        # [6,bs,100,4]
        # bbox的输出
        # 这里的bbox的值都是相对于图片的hw的因此都是小于1的，因此使用sigmoid将值约束在0-1
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # 6层decoder,取最后一层的输出
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # 计算辅助loss，计算前几层decoder输出的loss
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


# 计算Loss
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # 是一个list,每项内容都是需要计算的loss ['labels', 'boxes', 'cardinality']
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        # 正常类的系数就是1，最后一类的系数是0.1
        # 论文中说这种设置也类似于Faster R-CNN中的正负样本采样
        empty_weight[-1] = self.eos_coef  # no object calss的权重
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        分类loss
        indices 是一个list，长度是batch的大小
        indices的每一个item 是一个tuple，每个tuple里面是2个tensor，这两个tensor的长度相同
        tensor的长度就是target的数量
        第一个tensor表示的是100个框中的某个框的id
        第二个tensor表示的是这个框跟哪一个target中的box匹配，是target的box的id
        target 是要给list，长度是batch的大小
        target 是一个dict，包含了7项
        boxes,labels,image_id,area,iscrowd,orig_size,size
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs  # outputs 是detr网络的输出
        # 预测值
        src_logits = outputs['pred_logits']
        # 是两个tensor，第一个tensor是哪个图片的id，第二个tensor是分配的预测框的id
        idx = self._get_src_permutation_idx(indices)
        # 按照预测的顺序，组织一下gt的class的排序
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # 都用最后一类，背景类填充
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 分配上正确的gt的类别
        target_classes[idx] = target_classes_o
        # 预测值，目标值
        # 计算交叉熵
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # accuracy no_grad 并不是真正的loss，仅是为了最后输出用
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        # 这个不是一个真的loss
        # 计算每张图像预测的非no object的数量与真实GT的bbox的数量差
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']

        device = pred_logits.device

        # 每个图片gt的数量
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # 预测的不是最后一个背景类的数量
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        # 就是预测出是前景类的数目和真实的gt的数目的l1,就是差值
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        # 这个值可能是很大，比如90
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        # idx 是个tuple，正好将pred_boxes的第一维是batch的image，第二位是每个预测的box(每个图片100个),将这两个维度，直接取出来了
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # l1，没有使用smooth_l1
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # giou loss
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]) 类似这种，表明了预测的都是batch中那个image的id，用了cat拼接
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # tensor([23, 60, 37, 63, 78, 97, 16, 18, 21, 39, 42, 47, 61, 70, 77]) 表明了都是那些框体的id
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 只用计算mask loss时会调用
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # loss 是指定的某些loss的名称 ["labels", "boxes", "cardinality"] 以及masks
        # 计算各种loss的函数，一共就是这四种，每种对应了一个方法
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             经过transformer后的输出
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # without 这里排除aux
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # indices是list, 长度是bs, 每个item是一个tuple, 第一个值是100个框的id，第二个值是gt的id
        # 进行匈牙利匹配
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # batch中所有的box的数量
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)

        # 变成tensor
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            # 所有卡的num_boxes相同
            torch.distributed.all_reduce(num_boxes)  # todo all_reduce

        # 除以卡数
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算后是loss_ce, class_error, loss_bbox, loss_giou, cardinality_error
        # slef.losses 是labels,boxes, cardinality
        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            # 计算每一项的内容, 如 labels, boxes, cardinality
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 处理 aux loss
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:

            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                indices = self.matcher(aux_outputs, targets)

                for loss in self.losses:

                    if loss == 'masks':
                        # masks的loss就不计算这个了，计算量大
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}

                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)

                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}

                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        # 分数执行softmax
        prob = F.softmax(out_logits, -1)
        # 最大的分数，以及对应的label
        scores, labels = prob[..., :-1].max(-1)

        # 转换成两点坐标
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # 放大成正常的尺寸
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


# ----------------------------------------------------------------------------------------------------------------------

def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)
    # 创建backbone
    backbone = build_backbone(args)
    # 创建transformer
    transformer = build_transformer(args)
    # detr model
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    if args.masks:
        # 包装并代替了detr
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    # matcher 给 SetCriterion使用
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef}
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)


    # losses的内容
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    # bbox的后处理
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
