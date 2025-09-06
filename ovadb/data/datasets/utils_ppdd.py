from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# ppdd_categories_base = [
#     {"color": (220, 20, 60), "isthing":1, "id": 0, "name": "Reflective"},
#     {"color": (119, 11, 32), "isthing":1, "id": 1, "name": "Verti-Edge"},
#     {"color": (106, 0, 228), "isthing":1, "id": 2, "name": "Construction"}, # 4
#     {"color": (0, 60, 100), "isthing":1, "id": 3, "name": "Alligator"} # 5
# ]
ppdd_categories_base = [
    {"color": (220, 20, 60), "isthing":1, "id": 0, "name": "RC"},
    {"color": (119, 11, 32), "isthing":1, "id": 1, "name": "LEC"},
    {"color": (106, 0, 228), "isthing":1, "id": 2, "name": "CJC"}, # 4
    {"color": (0, 60, 100), "isthing":1, "id": 3, "name": "AC"} # 5
]
ppdd_categories_base_names = [x["name"] for x in ppdd_categories_base]

# ppdd_categories_novel = [
#     {"color": (0, 0, 142), "isthing":1, "id": 4, "name": "Corr-Shov-Disp"}, # 2
#     {"color": (0, 0, 230), "isthing":1, "id": 5, "name": "Rutt-Depress"} # 3
    
# ]
ppdd_categories_novel = [
    {"color": (0, 0, 142), "isthing":1, "id": 4, "name": "CSSC"}, # 2
    {"color": (0, 0, 230), "isthing":1, "id": 5, "name": "RDC"} # 3
    
]
ppdd_categories_novel_names = [x["name"] for x in ppdd_categories_novel]


# 클래스-속성 사전 category_id:attribute index
classid_to_attr = {
    1: ["geometry:thin and long", "geometry:perpendicular to the lane direction"],     # Reflective 1
    2: ["geometry:thin and long", "geometry:parallel to the lane direction", "spatial:pavement edge"],     # Verti-Edge 2
    3: ["geometry:perpendicular to the lane direction", "geometry:crescent-shaped", "geometry:curved", "geometry:horseshoe-shaped", "texture:wave-like"],     # Corr-Shov-Disp 3
    4: ["geometry:depressed", "geometry:being lower", "texture:widely fragmented and finely broken", "spatial:aligned with wheel paths"],     # Rutt-Depress 4
    12: ["geometry:straight", "geometry:thin and long", "geometry:parallel to the lane direction", "spatial:aligned with lane markings"],    # Construction 12
    13: ["geometry:polygonal", "texture:finely cracked", "texture:finely fragmented", "texture:alligator skin"],   # Alligator 13
}