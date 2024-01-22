#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "Tokenizer.hpp"
#include "string_utility.hpp"

using namespace cv;
using namespace std;
using namespace Ort;

struct Object
{
	cv::Rect box;
	string text;
	float prob;
};

static inline float sigmoid(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

class GroundingDINO
{
public:
	GroundingDINO(string modelpath, float box_threshold, string vocab_path, float text_threshold, bool with_logits);
	vector<Object> detect(Mat srcimg, string text_prompt);

private:
	void preprocess(Mat img);
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };
	const int size[2] = { 1200, 800 };  ////(宽度, 高度)

	std::shared_ptr<TokenizerBase> tokenizer;
	bool load_tokenizer(std::string vocab_path);

	std::vector<float> input_img;
	std::vector<std::vector<int64>> input_ids;
	std::vector<std::vector<uint8_t>> attention_mask;
	std::vector<std::vector<int64>> token_type_ids;
	std::vector<std::vector<uint8_t>> text_self_attention_masks;
	std::vector<std::vector<int64>> position_ids;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "GroundingDINO");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	const char* input_names[6] = { "img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask" };
	const char* output_names[2] = { "logits", "boxes" };
	
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	
	float box_threshold;
	float text_threshold;
	bool with_logits;
	const int max_text_len = 256;
	const char* specical_texts[4] = { "[CLS]", "[SEP]", ".", "?" };
	vector<int64> specical_tokens = { 101, 102, 1012, 1029 };  ///已经是定值了，不需要每张图片重新计算
};

GroundingDINO::GroundingDINO(string modelpath, float box_threshold, string vocab_path, float text_threshold, bool with_logits)
{
	std::wstring widestr = std::wstring(modelpath.begin(), modelpath.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);

	this->load_tokenizer(vocab_path);
	this->box_threshold = box_threshold;
	this->text_threshold = text_threshold;
	this->with_logits = with_logits;
}

bool GroundingDINO::load_tokenizer(std::string vocab_path)
{
	tokenizer.reset(new TokenizerClip);
	return tokenizer->load_tokenize(vocab_path);
}

void GroundingDINO::preprocess(Mat img)
{
	Mat rgbimg;
	cvtColor(img, rgbimg, COLOR_BGR2RGB);
	resize(rgbimg, rgbimg, cv::Size(this->size[0], this->size[1]));
	vector<cv::Mat> rgbChannels(3);
	split(rgbimg, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0* std[c]), (0.0 - mean[c]) / std[c]);
	}

	const int image_area = this->size[0] * this->size[1];
	this->input_img.resize(3 * image_area);
	size_t single_chn_size = image_area * sizeof(float);
	memcpy(this->input_img.data(), (float*)rgbChannels[0].data, single_chn_size);
	memcpy(this->input_img.data() + image_area, (float*)rgbChannels[1].data, single_chn_size);
	memcpy(this->input_img.data() + image_area * 2, (float*)rgbChannels[2].data, single_chn_size);
}

vector<Object> GroundingDINO::detect(Mat srcimg, string text_prompt)
{
	this->preprocess(srcimg);
	const int srch = srcimg.rows, srcw = srcimg.cols;
	
	std::transform(text_prompt.begin(), text_prompt.end(), text_prompt.begin(), ::tolower); ////转小写
	string caption = strip(text_prompt); ////去掉首尾空格符
	if (endswith(caption, ".") == 0)
	{
		caption += " .";
	}

	this->input_ids.resize(1); ////输入提示词是一个字符串，不再是数组
	this->attention_mask.resize(1);
	this->token_type_ids.resize(1);
	std::vector<int64> ids;
	tokenizer->encode_text(caption, ids);
	int len_ids = ids.size();
	int trunc_len = len_ids <= this->max_text_len ? len_ids : this->max_text_len;
	input_ids[0].resize(trunc_len);
	token_type_ids[0].resize(trunc_len);
	attention_mask[0].resize(trunc_len);
	for (int i = 0; i < trunc_len; i++)
	{
		input_ids[0][i] = ids[i];
		token_type_ids[0][i] = 0;
		attention_mask[0][i] = ids[i] > 0 ? 1 : 0;
	}
	////generate_masks_with_special_tokens_and_transfer_map
	////const int bs = input_ids.size();
	const int num_token = input_ids[0].size();
	vector<int> idxs;  ///输入是一个字符串，bs始终是1,因此这里不定义成二维的
	for (int i = 0; i < num_token; i++)
	{
		for (int j = 0; j < this->specical_tokens.size(); j++)
		{
			if (input_ids[0][i] == this->specical_tokens[j])
			{
				idxs.push_back(i);
			}
		}
	}
	
	len_ids = idxs.size();
	trunc_len = num_token <= this->max_text_len ? num_token : this->max_text_len;
	text_self_attention_masks.resize(1);
	text_self_attention_masks[0].resize(trunc_len*trunc_len);
	position_ids.resize(1);
	position_ids[0].resize(trunc_len);
	for (int i = 0; i < trunc_len; i++)
	{
		for (int j = 0; j < trunc_len; j++)
		{
			text_self_attention_masks[0][i*trunc_len + j] = (i == j ? 1 : 0);   ////对角线矩阵
		}
		position_ids[0][i] = 0;
	}
	int previous_col = 0;
	for (int i = 0; i < len_ids; i++)
	{
		const int col = idxs[i];
		if (col == 0 || col == num_token - 1)
		{
			text_self_attention_masks[0][col*trunc_len + col] = true;
			position_ids[0][col] = 0;
		}
		else
		{
			for (int j = previous_col + 1; j <= col; j++)
			{
				for (int k = previous_col + 1; k <= col; k++)
				{
					text_self_attention_masks[0][j*trunc_len + k] = true;
				}
				position_ids[0][j] = j - previous_col - 1;

			}
		}
		previous_col = col;
	}

	const int seq_len = input_ids[0].size();
	std::vector<int64_t> input_img_shape = { 1, 3, this->size[1], this->size[0] };
	std::vector<int64_t> input_ids_shape = { 1, seq_len };
	std::vector<int64_t> text_token_mask_shape = { 1, seq_len, seq_len };

	std::vector<Ort::Value> inputTensors;
	inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler,input_img.data(), input_img.size(), input_img_shape.data(), input_img_shape.size())));
	inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, input_ids[0].data(), input_ids[0].size(), input_ids_shape.data(), input_ids_shape.size())));
	inputTensors.push_back((Ort::Value::CreateTensor<bool>(memory_info_handler, reinterpret_cast<bool *>(attention_mask[0].data()), attention_mask[0].size(), input_ids_shape.data(), input_ids_shape.size())));  ///需要注意的是使用bool型，需要从uint_8的vector转为bool型, bool类型的vector没有data成员的
	inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, position_ids[0].data(), position_ids[0].size(), input_ids_shape.data(), input_ids_shape.size())));
	inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, token_type_ids[0].data(), token_type_ids[0].size(), input_ids_shape.data(), input_ids_shape.size())));
	inputTensors.push_back((Ort::Value::CreateTensor<bool>(memory_info_handler, reinterpret_cast<bool *>(text_self_attention_masks[0].data()), text_self_attention_masks[0].size(), text_token_mask_shape.data(), text_token_mask_shape.size())));  ///需要注意的是使用bool型，需要从uint_8的vector转为bool型, bool类型的vector没有data成员的

	Ort::RunOptions runOptions;
	vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names, inputTensors.data(), inputTensors.size(), this->output_names, 2);

	const float *ptr_logits = ort_outputs[0].GetTensorMutableData<float>();
	std::vector<int64_t> logits_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const float *ptr_boxes = ort_outputs[1].GetTensorMutableData<float>(); ////cx,cy,w,h
	std::vector<int64_t> boxes_shape = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	const int outh = logits_shape[1];
	const int outw = logits_shape[2];
	vector<int> filt_inds;
	vector<float> scores;
	for (int i = 0; i < outh; i++)  ///忽律掉第0维batch_size
	{
		float max_data = 0;
		for (int j = 0; j < outw; j++)
		{
			float x = sigmoid(ptr_logits[i*outw + j]);
			if (max_data < x)
			{
				max_data = x;
			}
		}
		if (max_data > this->box_threshold)
		{
			filt_inds.push_back(i);
			scores.push_back(max_data);
		}
	}

	std::vector<Object> objects;
	for (int i = 0; i < filt_inds.size(); i++)
	{
		////get_phrases_from_posmap
		const int ind = filt_inds[i];
		const int left_idx = 0, right_idx = 255;
		for (int j = left_idx+1; j < right_idx; j++)
		{
			float x = sigmoid(ptr_logits[ind*outw + j]);
			if (x > this->text_threshold)
			{
				const int64 token_id = input_ids[0][j];
				Object obj;
				obj.text = this->tokenizer->tokenizer_idx2token[token_id];  
				obj.prob = scores[i];
				int xmin = int((ptr_boxes[ind * 4] - ptr_boxes[ind * 4 + 2] * 0.5)*srcw);
				int ymin = int((ptr_boxes[ind * 4 + 1] - ptr_boxes[ind * 4 + 3] * 0.5)*srch);
				///int xmax = int((ptr_boxes[ind * 4] + ptr_boxes[ind * 4 + 2] * 0.5)*srcw);
				///int ymax = int((ptr_boxes[ind * 4 + 1] + ptr_boxes[ind * 4 + 3] * 0.5)*srch);
				int w = int(ptr_boxes[ind * 4 + 2] * srcw);
				int h = int(ptr_boxes[ind * 4 + 3] * srch);
				obj.box = Rect(xmin, ymin, w, h);
				objects.push_back(obj);

				break;  ///只有一个元素，提前结束for循环
			}
		}
	}
	return objects;
}


int main()
{
	GroundingDINO mynet("weights/groundingdino_swint_ogc.onnx", 0.3, "vocab.txt", 0.25, true);   ////加载groundingdino_swinb_cogcoor.onnx出现异常,但是python的可以正常推理,使用都是onnxruntime1.14.1版本的

	const std::string imgpath = "images/cat_dog.jpeg";
	std::string text_prompt = "chair . person . dog .";   ////每个类别名称之间以 . 隔开
	Mat srcimg = imread(imgpath);
	vector<Object> objects = mynet.detect(srcimg, text_prompt);

	for (size_t i = 0; i < objects.size(); i++)
	{
		cv::rectangle(srcimg, objects[i].box, cv::Scalar(0, 0, 255), 2);
		string label = format("%.2f", objects[i].prob);
		label = objects[i].text + ":" + label;
		cv::putText(srcimg, label, cv::Point(objects[i].box.x, objects[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
	}

	///imwrite("result.jpg", srcimg);
	static const string kWinName = "GroundingDINO use OnnxRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();

	return 0;
}