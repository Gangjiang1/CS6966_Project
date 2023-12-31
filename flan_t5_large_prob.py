import transformers
import os
# import torch
# 选择的model
# base 2.2b
# large 7.7b
model_checkpoint = "google/flan-t5-large"
import torch
from datasets import load_dataset

#raw_datasets = load_dataset("json", data_files="10.17_geo_explain_1w.json")
#raw_datasets = load_dataset("json", data_files="11.3_geo_load_1w.json")
raw_datasets = load_dataset("json", data_files="11.3_geo_load_1w.json")
test_datasets = load_dataset("json", data_files="11.3_geo_load_100_test.json")
print (raw_datasets)

print (raw_datasets["train"][0])

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint,return_dict_in_generate=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# new_tokens = ['\n', ' ']
# num_added_tokens = tokenizer.add_tokens(new_tokens)
# print('Number of tokens added:', num_added_tokens)
# model.resize_token_embeddings(len(tokenizer))
# # 保存tokenizer 供以后使用
# tokenizer.save_pretrained('new_tokenizer')


# 定义要生成的文本
text = raw_datasets["train"][0]["Idf"]

# 使用标记器将文本转换为标记
tokens = tokenizer(text, return_tensors="pt")

# 计算标记的数量
token_count = len(tokens["input_ids"][0])

print("token_count:", token_count)


if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "Simulation: "
else:
    prefix = ""

max_input_length = 128
# max_target_length = 5000

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["Prompt"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    # labels = tokenizer(text_target=examples["Idf"], max_length=max_target_length, truncation=True)
    labels = tokenizer(text_target=examples["Idf"], truncation=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# 这里 将输出做了处理，然后也添加到了 input中

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
tokenized_test_datasets = test_datasets.map(preprocess_function, batched=True)
print(tokenized_datasets)
print(tokenized_test_datasets)

batch_size = 5
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}_999Eplus-LLM",
    # evaluation_strategy = "epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    weight_decay=0,
    # save_strategy="no",
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# 这个类提供了一种将输入序列（例如源文本）和目标序列（例如目标文本或生成的文本）组合成模型所需格式的方法。
# 它处理了一些与序列长度、填充和截断相关的细节，以确保数据对模型的输入和输出具有一致的形状。

# 并行
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)
# 用这个会导致size报错，不知道是不是因为size太大

trainer = Seq2SeqTrainer(
# trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    args=args,
    data_collator=data_collator
)


print ("Parameters:",'name:flan t5 large','dataset:1w_explain','token size:', token_count,'epochs:', 1, 'batch size:', batch_size)
trainer.train()
# 准备测试文本
# small_test_dataset = tokenized_test_datasets["train"].shuffle(seed=142).select(range(1))
small_test_dataset = tokenized_test_datasets["train"].shuffle(seed=142).select(range(10))
# small_test_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1))
test_prompt = small_test_dataset['Prompt']
print ('Test prompt: ', test_prompt)
test_idf = small_test_dataset["Idf"]
print ('Test idf: ', test_idf)
# text_idf

input_text = test_prompt
print ("Input: ", input_text)

# generation_config = model.generation_config
# generation_config.min_new_tokens = token_count-100
# generation_config.max_new_tokens = token_count+100
# generation_config.temperature = 0.1
# generation_config.top_p = 0.1
# generation_config.num_return_sequences = 1
# generation_config.pad_token_id = tokenizer.eos_token_id
# generation_config.eos_token_id = tokenizer.eos_token_id

device = "cuda:0"
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=False).input_ids.to(device)
# model.eval()
# 使用模型预测
generated_outputs = model.generate(input_ids,
    min_new_tokens = token_count-100,
    max_new_tokens = token_count+100,
    do_sample=True,
    num_return_sequences=1,
    output_scores=True,
    return_dict_in_generate=True,
    pad_token_id = tokenizer.eos_token_id,
    eos_token_id = tokenizer.eos_token_id,
    temperature = 0.5,
    top_p = 0.5,
)

gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]
probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)

gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
print("每行的最大值:")
print(gen_probs)

# -> shape [3, 15, vocab_size] 最后一个维度进行softmax
# 求出每行的最大值
max_values, _ = torch.max(probs, dim=2)
# 打印每行的最大值
print("每行的最大值:")
print(max_values)
print(max_values[0,:])
# 使用模型解码输出
# print('Output: ', tokenizer.decode(outputs[0], skip_special_tokens=True))
generated_text = tokenizer.decode(generated_outputs[0][0], skip_special_tokens=True)
generated_text = generated_text.replace("_", " ")
# generated_text = generated_text.replace("<", 4*" ")
# generated_text = generated_text.replace("<", "\t")
generated_text = generated_text.replace("|", "\n")
print("Generated_text>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(generated_text)

# 打开一个文本文件用于写入，如果文件不存在则创建它
with open("generated_flan_t5_large_with load.txt", "w") as file:
    # 将生成的文本写入文件
    #file.write(input_text)
    # file.write('\n')
    # file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # file.write('\n')
    file.write(generated_text)
    # 将 max_values 转换为字符串并写入文件
    file.write("\n")
    max_values_str = str(max_values[0, :].tolist())
    file.write(max_values_str)
    file.write("\n")
    max_gen_probs_str = str(gen_probs.tolist())
    file.write(max_gen_probs_str)
print("Generated text saved to generated_text.txt")