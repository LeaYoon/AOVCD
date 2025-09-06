import os
import numpy as np
import torch
from torch import nn
import dill as pickle
from torch.nn import functional as F
from detectron2.config import configurable
from typing import Dict, List, Optional, Tuple
from detectron2.layers import Linear, ShapeSpec
from detectron2.structures import Boxes, Instances

from ovadb.modeling.logged_module import LoggedModule

"""
Similar to ZeroShotClassifier but including atttributes
"""

class PPDDAttributeClassifier(LoggedModule):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_attributes: int,
        attr_weight_path: str,
        attr_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        remap_category_id=False, 
        remap_category_map=None
    ):
        super().__init__()
        if isinstance(input_shape, int):  # for compatibility
            input_shape = ShapeSpec(channels=input_shape)

        input_size = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.use_bias = use_bias < 0
        self.zs_weight_dim = attr_weight_dim
        
        self.remap_category_id = remap_category_id
        self.remap_category_map = remap_category_map

        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, attr_weight_dim)

        if attr_weight_path == "rand":
            attr_weight = torch.randn((attr_weight_dim, num_attributes))
            nn.init.normal_(attr_weight, std=0.01)
        else:
            attr_weight = (
                torch.tensor(np.load(attr_weight_path), dtype=torch.float32)
                .permute(1, 0)
                .contiguous()
            )  # (D x A)

        if norm_weight:
            attr_weight = F.normalize(attr_weight, p=2, dim=0)

        self.register_buffer("attr_weight", attr_weight)

        assert self.attr_weight.shape[1] == num_attributes, self.attr_weight.shape

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "num_attributes": cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES,
            "attr_weight_path": cfg.MODEL.ROI_BOX_HEAD.ATTRIBUTE_WEIGHT_PATH,
            "attr_weight_dim": cfg.MODEL.ROI_BOX_HEAD.ATTRIBUTE_WEIGHT_DIM,
            "use_bias": cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            "norm_weight": cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            "norm_temperature": cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            "remap_category_id": cfg.INPUT.REMAP_CATEGORY_ID,
            "remap_category_map": cfg.INPUT.REMAP_CATEGORY_MAP, 
        }

    # def forward(self, x, attr_weight_override=None):
    #     x = self.linear(x)
    #     attr_weight = attr_weight_override if attr_weight_override is not None else self.attr_weight
    #     if self.norm_weight:
    #         x = self.norm_temperature * F.normalize(x, p=2, dim=1)
    #         attr_weight = F.normalize(attr_weight, p=2, dim=0)
    #     x = torch.mm(x, attr_weight)
    #     if self.use_bias:
    #         x = x + self.cls_bias
    #     return x
    def forward(self, x, second_arg=None):

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # inference
        if isinstance(second_arg, list) and isinstance(second_arg[0], Instances):
            instances = second_arg
            attr_weight = self.attr_weight
            if self.norm_weight:
                x_noun = self.norm_temperature * F.normalize(x, p=2, dim=1)
                x_attr = self.norm_temperature * F.normalize(x, p=2, dim=1)
            else:
                x_noun = x
                x_attr = x

            x_nouns = self.noun_pred(x_noun)
            x_attrs = self.attr_pred(x_attr)

            if hasattr(self, "att_syn_len") and self.num_attributes != len(self.att_ids):
                x_attrs_syn = x_attrs.split(self.att_syn_len, dim=1)
                x_attrs = torch.stack(
                    [x_syn.max(axis=1)[0] for x_syn in x_attrs_syn], dim=1
                )
            attr_prob = x_attrs.sigmoid()
            noun_prob = x_nouns.softmax(dim=-1)

            num_inst_per_image = [len(p) for p in instances]
            attr_prob_split = attr_prob.split(num_inst_per_image, dim=0)
            noun_prob_split = noun_prob.split(num_inst_per_image, dim=0)    
            for p_inst, p_attr, p_noun in zip(instances, attr_prob_split, noun_prob_split):
                p_inst.att_scores = p_attr
                p_inst.noun_scores = p_noun
            return instances

        elif isinstance(second_arg, torch.Tensor) or second_arg is None:
            x = self.linear(x)
            attr_weight = second_arg if second_arg is not None else self.attr_weight
            if self.norm_weight:
                x = self.norm_temperature * F.normalize(x, p=2, dim=1)
                attr_weight = F.normalize(attr_weight, p=2, dim=0)
            x = torch.mm(x, attr_weight)
            if self.use_bias:
                x = x + self.cls_bias
            return x
        else:
            raise ValueError(
                "Second argument must be either a tensor or a list of Instances, got {}".format(
                    type(second_arg)
                )
            )
        
    
    def set_embeddings(self, path_weights, is_noun=True, zs_weight=None):
    
        assert (
            os.path.isfile(path_weights) or zs_weight is not None
        ), "Path to classification weights must be valid: {}".format(path_weights)

        # get weights
        # device = self.noun_pred.weight.device
        device = next(self.parameters()).device
        if os.path.isfile(path_weights):
            print("Loading {} for attribute head".format(path_weights))
            # if saved as numpy - synonyms are average
            if path_weights.endswith(".npy"):
                zs_weight = torch.tensor(
                    np.load(path_weights), dtype=torch.float32
                )  # C x D
                self.att_ids = list(range(zs_weight.shape[0]))
                self.att_syn_len = [1] * zs_weight.shape[0]
                self.num_attributes = zs_weight.shape[0]
            # saved as pickle
            elif path_weights.endswith(".pkl"):
                att_syn_dict = pickle.load(open(path_weights, "rb"))
                self.att_syn_len = att_syn_dict["syn_len"]
                self.att_ids = att_syn_dict["ids"]
                self.num_attributes = len(self.att_syn_len)
                zs_weight = torch.tensor(
                    att_syn_dict["feat"], dtype=torch.float32
                )  # C x D
        assert zs_weight is not None, "No zs_weight provided for {}".format(path_weights)
        assert zs_weight.shape[1] == self.linear.out_features, "The zs_weight dimension {} has to match the one saved in the model {}".format(
            zs_weight.shape[1], self.linear.out_features
        )   
        if torch.is_tensor(zs_weight):
            zs_weight = zs_weight.clone().detach().to(device)
        else:
            zs_weight = torch.tensor(zs_weight, device=device)

        # assert (
        #     zs_weight.shape[1] == self.zs_weight_dim
        # ), "The weigts dimension {} has to match the one saved in the model {}".format(
        #     zs_weight.shape[1], self.zs_weight_dim
        # )

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=1)

        # self.log("zs_weight", zs_weight)

        # noun
        if is_noun:
            self.num_classes = zs_weight.shape[0]
            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((1, self.zs_weight_dim))], dim=0
            )  # (C + 1) x D
            self.noun_pred = nn.Linear(self.zs_weight_dim, self.num_classes + 1)
            self.noun_pred.weight.data = zs_weight
            if self.use_bias:
                self.noun_pred.bias.data = (
                    torch.ones_like(self.noun_pred.bias.data, device=device) * self.bias
                )
            else:
                self.noun_pred.bias.data = torch.zeros_like(
                    self.noun_pred.bias.data, device=device
                )
        
        # # attr
        if not is_noun:
            self.attr_pred = nn.Linear(self.zs_weight_dim, zs_weight.shape[0])
            self.attr_pred.weight.data = zs_weight
            if self.use_bias:
                self.attr_pred.bias.data = (
                    torch.ones_like(self.attr_pred.bias.data, device=device) * self.bias
                )
            else:
                self.attr_pred.bias.data = torch.zeros_like(
                    self.attr_pred.bias.data, device=device
                )

        self.freeze_classifiers(is_noun)

    #     if is_noun:
    #         self.attr_weight = zs_weight.new_zeros(1, zs_weight.shape[1])
    #         self.num_classes = zs_weight.shape[0]
    #         self.register_buffer("zs_noun_weight", zs_weight)
    #     else:
    #         self.num_attributes = zs_weight.shape[0]
    #         self.register_buffer("attr_weight", zs_weight)

    #     self.attribute_on = True
    
    # @property
    # def num_classes(self):
    #     return getattr(self, "num_classes", self.attr_weight.shape[1])

    # @property
    # def num_attributes(self):
    #     return getattr(self, "num_attributes", self.attr_weight.shape[0])

    def freeze_classifiers(self, is_noun=True):
        if is_noun:
            self.noun_pred.weight.requires_grad = False
            self.noun_pred.bias.requires_grad = False
        if not is_noun:
            self.attr_pred.weight.requires_grad = False
            self.attr_pred.bias.requires_grad = False

class AttributeClassifier(LoggedModule):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        num_attributes: int,
        att_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        norm_temp_att: float = 50.0,
        use_sigmoid_ce: bool = False,
        add_feature: bool = False,
        conditional_prediction: bool = False,
        conditional_weights: str = "",
        conditional_obj_label: bool = False,
        norm_temp_cond_att: float = 50.0
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.norm_temp_att = norm_temp_att
        self.zs_weight_dim = zs_weight_dim
        self.use_sigmoid_ce = use_sigmoid_ce
        self.use_bias = use_bias < 0
        self.bias = use_bias

        # Add the embedding based layer
        # nouns
        self.num_classes = num_classes
        self.noun_pred = nn.Linear(self.zs_weight_dim, self.num_classes)
        nn.init.normal_(self.noun_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.noun_pred.bias, 0)

        zs_weight = torch.randn((num_classes, zs_weight_dim))
        nn.init.normal_(zs_weight, std=0.01)
        self.set_embeddings(zs_weight_path, is_noun=True, zs_weight=zs_weight)

        # attributes
        self.num_attributes = num_attributes
        self.attr_pred = nn.Linear(self.zs_weight_dim, self.num_attributes)
        nn.init.normal_(self.attr_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.attr_pred.bias, 0)
        self.att_syn_len = [1] * num_attributes
        self.att_ids = list(range(num_attributes))

        zs_weight = torch.randn((num_attributes, zs_weight_dim))
        nn.init.normal_(zs_weight, std=0.01)
        self.set_embeddings(att_weight_path, is_noun=False, zs_weight=zs_weight)

        self.add_feature = add_feature

        self.conditional_prediction = conditional_prediction
        if conditional_prediction:
            self.set_conditional_embeddings(
                conditional_weights, num_classes, num_attributes
            )
            self.use_label = conditional_obj_label
            self.norm_temp_cond_att = norm_temp_cond_att

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "zs_weight_path": cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            "zs_weight_dim": cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            "use_bias": cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            "norm_weight": cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            "norm_temperature": cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            "norm_temp_att": cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP_ATTRIBUTE,
            "num_attributes": cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES,
            "att_weight_path": cfg.MODEL.ROI_BOX_HEAD.ATTRIBUTE_WEIGHT_PATH,
            "use_sigmoid_ce": cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            "add_feature": cfg.MODEL.ROI_BOX_HEAD.ADD_BOX_FEATURES_PREDICTION,
            "conditional_prediction": cfg.EVALUATION_ATTRIBUTE.CONDITIONAL,
            "conditional_weights": cfg.EVALUATION_ATTRIBUTE.EMBEDDING_DICTIONARY,
            "conditional_obj_label": cfg.EVALUATION_ATTRIBUTE.CONDITIONAL_USE_OBJ_LABEL,
            "norm_temp_cond_att": cfg.EVALUATION_ATTRIBUTE.CONDITIONAL_TEMPERATURE,
        }

    def set_embeddings(self, path_weights, is_noun=True, zs_weight=None):
        assert (
            os.path.isfile(path_weights) or zs_weight is not None
        ), "Path to classification weights must be valid: {}".format(path_weights)

        # get weights
        device = self.noun_pred.weight.device
        if os.path.isfile(path_weights):
            print("Loading {} for attribute head".format(path_weights))
            # if saved as numpy - synonyms are average
            if path_weights.endswith(".npy"):
                zs_weight = torch.tensor(
                    np.load(path_weights), dtype=torch.float32
                )  # C x D
                self.att_ids = list(range(zs_weight.shape[0]))
                self.att_syn_len = [1] * zs_weight.shape[0]
                self.num_attributes = zs_weight.shape[0]
            # saved as pickle
            elif path_weights.endswith(".pkl"):
                att_syn_dict = pickle.load(open(path_weights, "rb"))
                self.att_syn_len = att_syn_dict["syn_len"]
                self.att_ids = att_syn_dict["ids"]
                self.num_attributes = len(self.att_syn_len)
                zs_weight = torch.tensor(
                    att_syn_dict["feat"], dtype=torch.float32
                )  # C x D
        if torch.is_tensor(zs_weight):
            zs_weight = zs_weight.clone().detach().to(device)
        else:
            zs_weight = torch.tensor(zs_weight, device=device)


        assert (
            zs_weight.shape[1] == self.zs_weight_dim
        ), "The weigts dimension {} has to match the one saved in the model {}".format(
            zs_weight.shape[1], self.zs_weight_dim
        )

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=1)

        self.log("zs_weight", zs_weight)

        # noun
        if is_noun:
            self.num_classes = zs_weight.shape[0]
            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((1, self.zs_weight_dim))], dim=0
            )  # (C + 1) x D
            self.noun_pred = nn.Linear(self.zs_weight_dim, self.num_classes + 1)
            self.noun_pred.weight.data = zs_weight
            if self.use_bias:
                self.noun_pred.bias.data = (
                    torch.ones_like(self.noun_pred.bias.data, device=device) * self.bias
                )
            else:
                self.noun_pred.bias.data = torch.zeros_like(
                    self.noun_pred.bias.data, device=device
                )

        # attr
        if not is_noun:
            self.attr_pred = nn.Linear(self.zs_weight_dim, zs_weight.shape[0])
            self.attr_pred.weight.data = zs_weight
            if self.use_bias:
                self.attr_pred.bias.data = (
                    torch.ones_like(self.attr_pred.bias.data, device=device) * self.bias
                )
            else:
                self.attr_pred.bias.data = torch.zeros_like(
                    self.attr_pred.bias.data, device=device
                )

        self.freeze_classifiers(is_noun)

    def freeze_classifiers(self, is_noun=True):
        if is_noun:
            self.noun_pred.weight.requires_grad = False
            self.noun_pred.bias.requires_grad = False
        if not is_noun:
            self.attr_pred.weight.requires_grad = False
            self.attr_pred.bias.requires_grad = False

    def set_conditional_embeddings(
        self,
        conditional_weights_path,
        num_classes,
        num_attributes,
        idCls2cls=None,
        idAtt2att=None,
    ):
        assert os.path.isfile(
            conditional_weights_path
        ), "Path to classification weights must be valid: {}".format(
            conditional_weights_path
        )
        self.dict_noun_att = pickle.load(open(conditional_weights_path, "rb"))
        if idCls2cls is not None:
            self.idCls2cls = idCls2cls
        else:
            self.idCls2cls = {}
        for key, val in self.dict_noun_att.items():
            zs_weight = torch.tensor(np.asarray(val), dtype=torch.float32)
            if self.norm_weight:
                zs_weight = F.normalize(zs_weight, p=2, dim=1)
            self.dict_noun_att[key] = zs_weight
            if key not in self.idCls2cls.keys():
                self.idCls2cls[len(self.idCls2cls)] = key

    def forward(
        self, x, instances: List[Instances], classifier=None, attribute_cls=None
    ):
        """
        Inputs:
            x: N x D
            per-region features of shape (N, ...) for N bounding boxes to predict.
            classifier: (Cn x D)
            attribute_cls: (Ca x D)
        """

        if classifier is not None:
            self.set_embeddings("", is_noun=True, zs_weight=classifier)
        if attribute_cls is not None:
            self.set_embeddings("", is_noun=False, zs_weight=attribute_cls)

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if self.norm_weight:
            x_noun = self.norm_temperature * F.normalize(x, p=2, dim=1)
            x_attr = self.norm_temp_att * F.normalize(x, p=2, dim=1)
        else:
            x_noun = x
            x_attr = x

        x_nouns = self.noun_pred(x_noun)
        x_attrs = self.attr_pred(x_attr)

        # take max over att synonyms
        if self.num_attributes != len(self.att_ids):
            # split into synonyms
            x_attrs_syn = x_attrs.split(self.att_syn_len, dim=1)
            # take arg max
            x_attrs_maxsyn = []
            x_attrs_idxsyn = []
            for x_syn in x_attrs_syn:
                xmax_val, xmax_idx = x_syn.max(axis=1)
                x_attrs_maxsyn.append(xmax_val)
                x_attrs_idxsyn.append(xmax_idx)
            x_attrs = torch.stack(x_attrs_maxsyn, axis=1)

        instances = self.attribute_rcnn_inference(x, x_nouns, x_attrs, instances)

        if self.add_feature:
            num_inst_per_image = [len(p) for p in instances]
            features = x.split(num_inst_per_image, dim=0)
            for feature, instance in zip(features, instances):
                instance.features = feature

        return instances

    def predict_probs(self, scores, proposals):
        """
        support sigmoid
        """
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_att_prob(self, scores, proposals):
        """
        applies sigmoid
        """
        num_inst_per_image = [len(p) for p in proposals]
        probs = scores.sigmoid()
        return probs.split(num_inst_per_image, dim=0)

    def pred_conditional_prob(self, x, instance):
        if self.norm_weight:
            x = self.norm_temp_cond_att * F.normalize(x, p=2, dim=1)

        x_cond_attrs = []
        for box_idx in range(len(instance)):
            # get index of object
            if self.use_label:
                # TODO: get labels until this point
                assert instance.has(
                    "gt_classes"
                ), "Instance does not have gt_classes to do conditional prediction"
                noun_idx = instance.gt_classes[box_idx].item()
            else:
                # use predicted labels
                noun_idx = instance.pred_classes[box_idx].item()

            atts_vector = self.dict_noun_att[self.idCls2cls[noun_idx]].to(x.device)
            x_cond_attr = torch.mm(x[box_idx : box_idx + 1], atts_vector.T)
            x_cond_attrs.append(x_cond_attr)

        if len(x_cond_attrs) > 0:
            x_cond_attrs = torch.cat(x_cond_attrs, axis=0)

            if self.use_bias:
                x_cond_attrs = x_cond_attrs + self.cls_bias

            cond_attr_prob = self.predict_att_prob(x_cond_attrs, [[0] * len(x)])[0]
            instance.cond_att_scores = cond_attr_prob
        else:
            instance.cond_att_scores = torch.zeros(0).to(self.noun_pred.weight.device)

    def attribute_rcnn_inference(
        self,
        x,
        x_nouns: torch.Tensor,
        x_attrs: torch.Tensor,
        instances: List[Instances],
    ):
        noun_prob = self.predict_probs(x_nouns, instances)
        attr_prob = self.predict_att_prob(x_attrs, instances)

        for noun, att, instance in zip(noun_prob, attr_prob, instances):
            instance.noun_scores = noun
            instance.att_scores = att

            if self.conditional_prediction:
                self.pred_conditional_prob(x, instance)

        return instances
