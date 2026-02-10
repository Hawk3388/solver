import re
import ast
from PIL import Image, ImageDraw, ImageFont
import ollama

def parse_grounding_output(output_text):
    """
    Extrahiert eine Liste von Einträgen:
    [
       {'label': 'text', 'boxes': [[x1,y1,x2,y2], ...]},
       ...
    ]
    """
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = re.findall(pattern, output_text, re.DOTALL)

    results = []
    for label, box_string in matches:
        boxes = ast.literal_eval(box_string)
        results.append({
            "label": label,
            "boxes": boxes
        })
    return results


def scale_boxes_0_999_to_pixels(boxes, w, h):
    scaled = []
    for x1, y1, x2, y2 in boxes:
        scaled.append([
            int(x1 / 999 * w),
            int(y1 / 999 * h),
            int(x2 / 999 * w),
            int(y2 / 999 * h),
        ])
    return scaled


def draw_boxes_on_image(image_path, output_text, output_path="boxed.png"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    grounding = parse_grounding_output(output_text)

    # Colors → deterministic by label
    import random
    random.seed(42)
    color_map = {}

    for item in grounding:
        label = item["label"]

        if label not in color_map:
            color_map[label] = (
                random.randint(50,255),
                random.randint(50,255),
                random.randint(50,255),
            )

        color = color_map[label]

        # scale coords
        boxes = scale_boxes_0_999_to_pixels(item["boxes"], w, h)

        # draw
        for (x1,y1,x2,y2) in boxes:
            draw.rectangle([x1,y1,x2,y2], outline=color, width=4)
            draw.text((x1+5, y1+5), label, fill=color)

    img.save(output_path)
    print("Saved:", output_path)

if __name__ == '__main__':
    path = "arbeitsblatt.png"

    stream = True

    response = ""

    response_stream = ollama.generate(model="deepseek-ocr", prompt="""<|grounding|>\nLocate <|ref|>fillable answer fields<|/ref|> in the image and output them in bounding boxes.""", images=[path], stream=stream)

    if stream:
        for chunk in response_stream:
            chunk = chunk.response
            print(chunk, end='', flush=True)
            response += chunk
    else:
        response = response_stream.response

    draw_boxes_on_image(path, response)