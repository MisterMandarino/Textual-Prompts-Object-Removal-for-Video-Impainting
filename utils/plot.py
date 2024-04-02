import supervision as sv
import numpy as np

def plot_segmented_image(img, masks):
    box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    source_image = box_annotator.annotate(scene=img.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=img.copy(), detections=detections)

    sv.plot_images_grid(
        images=[source_image, segmented_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

def get_distance(box1, box2):

    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2

    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    x = abs(cx1 - cx2)
    y = abs(cy1 - cy2)
    distance = np.sqrt(x**2 + y**2)

    return distance