from .attribute_classifier import AttributeClassifier
from .attribute_classifier import PPDDAttributeClassifier

# def build_attribute_predictor(cfg, input_shape):
#     """
#     Build a attribute head.
#     """
#     return AttributeClassifier(cfg, input_shape)

def build_attribute_predictor(cfg, input_shape):
    """
    Build a PPDD attribute head.
    """
    return PPDDAttributeClassifier(cfg, input_shape)
