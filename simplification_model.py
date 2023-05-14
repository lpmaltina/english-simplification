class SimplificationModel:
    def __init__(self, tokenizer_type, model_type, model_dir, max_target_length=512):
        self.tokenizer = tokenizer_type.from_pretrained(model_dir)
        self.model = model_type.from_pretrained(model_dir)
        self.max_target_length = max_target_length

    def simplify(self, text):
        inputs = ["Simplify English: " + text]
        inputs = self.tokenizer(inputs, return_tensors="pt")
        output = self.model.generate(**inputs, num_beams=8, do_sample=True, max_length=self.max_target_length)
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return decoded_output
