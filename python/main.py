import argparse
import cv2
import numpy as np
import onnxruntime
from clip_tokenizer import tokenize, FullTokenizer, generate_masks_with_special_tokens_and_transfer_map

def resize_image(srcimg, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):  ###返回(高度, 宽度)
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
    size = get_size((img.shape[1], img.shape[0]), size, max_size)
    rescaled_image = cv2.resize(img, (size[1], size[0]))
    return rescaled_image

def get_phrases_from_posmap(posmap, input_ids, tokenizer, left_idx=0, right_idx=255):
    if posmap.ndim == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = np.nonzero(posmap)[0].tolist()
        if len(non_zero_idx)>0:
            token_ids = [input_ids[i] for i in non_zero_idx]
            return tokenizer.convert_ids_to_tokens(token_ids)[0]
            ##return ' '.join(tokenizer.convert_ids_to_tokens(token_ids))  ###原始是返回一段描述的,如果只是检测目标,可以只返回一个单词
        else:
            return None
    else:
        raise NotImplementedError("posmap must be 1-dim")

class GroundingDINO():
    def __init__(self, modelpath, box_threshold, vocab_path, text_threshold=None, with_logits=True):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(modelpath, so)  ###opencv-dnn读取失败
        # for inp in self.net.get_inputs():
        #     print(inp)
        # for oup in self.net.get_outputs():
        #     print(oup)

        self.input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]
        self.output_names=["logits", "boxes"]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.with_logits = with_logits
        
        self.size = [1200, 800]  ###(宽度, 高度)
        self.max_size = None
        # self.size = 800
        # self.max_size = 1333
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
        self.max_text_len = 256
        self.specical_texts = ["[CLS]", "[SEP]", ".", "?"]
        self.tokenizer = FullTokenizer(vocab_file=vocab_path)

    def detect(self, srcimg, text_prompt):
        rescaled_image = resize_image(srcimg, self.size, max_size=self.max_size)
        img = (rescaled_image.astype(np.float32) / 255.0 - self.mean_) / self.std_
        inputs = {"img":np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)}

        caption = text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + " ."
        
        input_ids, token_type_ids, attention_mask, specical_tokens = tokenize(self.tokenizer, caption, self.specical_texts, context_length=self.max_text_len)
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids, specical_tokens)
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]
        
            position_ids = position_ids[:, : self.max_text_len]
            input_ids = input_ids[:, : self.max_text_len]
            attention_mask = attention_mask[:, : self.max_text_len]
            token_type_ids = token_type_ids[:, : self.max_text_len]

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        inputs["token_type_ids"] = token_type_ids
        inputs["position_ids"] = position_ids
        inputs["text_token_mask"] = text_self_attention_masks 

        outputs = self.net.run(self.output_names, inputs)
        
        prediction_logits_ = np.squeeze(outputs[0], axis=0) #[0]  # prediction_logits.shape = (nq, 256)
        prediction_logits_ = 1/(1+np.exp(-prediction_logits_))
        
        prediction_boxes_ = np.squeeze(outputs[1], axis=0) #[0]  # prediction_boxes.shape = (nq, 4)

        filt_mask = np.max(prediction_logits_, axis=1) > self.box_threshold
        logits_filt = prediction_logits_[filt_mask]  # num_filt, 256
        boxes_filt = prediction_boxes_[filt_mask]  # num_filt, 4

        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, input_ids[0, :], self.tokenizer)
            if pred_phrase is None:
                continue
            if self.with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

def draw_boxes_to_image(image, boxes, labels):
    h,w = image.shape[:2]
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * np.array([w, h, w, h])
        # from xywh to xyxy
        box[:2] -= box[2:] * 0.5
        box[2:] += box[:2]

        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        # txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        # cv2.rectangle(image, (xmin, ymin + 1), (xmin + txt_size[0] + 1, ymin + int(1.5 * txt_size[1])), (255, 255, 255), -1)
        cv2.putText(image, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="weights/groundingdino_swint_ogc.onnx", help="onnx model path")
    parser.add_argument("--image_path", type=str, default="images/cat_dog.jpeg", help="path to image file")
    parser.add_argument("--text_prompt", type=str, default="chair . person . dog .", help="text prompt, 每个类别名称之间以 . 隔开")  ###cat_dog.jpeg的提示词:"chair . person . dog .""     demo7.jpg的提示词:"Horse . Clouds . Grasses . Sky . Hill ."
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    args = parser.parse_args()

    mynet = GroundingDINO(args.model_path, args.box_threshold, "vocab.txt", text_threshold=args.text_threshold)
    srcimg = cv2.imread(args.image_path)

    boxes_filt, pred_phrases = mynet.detect(srcimg, args.text_prompt)
    drawimg = draw_boxes_to_image(srcimg, boxes_filt, pred_phrases)

    # cv2.imwrite('result.jpg', drawimg)
    winName = 'GroundingDINO use OnnxRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
