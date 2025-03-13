import torch
import re
import torch.nn as nn


def load_RED_model(model):
    model = ActivationLLama(model)
    return model


class ActivationLayer(nn.Module):
    def __init__(self, hidden_size, update_layer, layer_type="all", op_position="ffn", is_llama=False):
        super().__init__()
        self.update_layer = update_layer
        self.layer_type = layer_type
        self.op_position = op_position
        if (is_llama):
            self.weight_type = torch.bfloat16
        else:
            self.weight_type = torch.float32
        if (self.layer_type == "all"):
            self.delta_vector = nn.ParameterDict({
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size)),
                "activation_bias": nn.Parameter(torch.zeros(1, hidden_size)),
            })
        elif (self.layer_type == "scaling"):
            self.delta_vector = nn.ParameterDict({
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size))
            })
        elif (self.layer_type == "bias"):
            self.delta_vector = nn.ParameterDict({
                "activation_bias": nn.Parameter(torch.zeros(1, hidden_size))
            })
        elif (self.layer_type == "ln"):
            self.delta_vector = nn.ParameterDict({
                "activation_ln": nn.LayerNorm(hidden_size),
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size)),
                "activation_bias": nn.Parameter(torch.zeros(1, hidden_size)),
            })

        self.weight = torch.rand(1)
        self.delta_vector.to(self.weight_type)

    def forward(self, x, input_tensor=None):
        if (self.op_position == "res" or self.op_position == "res_with_attn" or self.op_position == "res_with_res"):
            hidden_states = self.update_layer(x, input_tensor)
        else:
            hidden_states = self.update_layer(x)

        if (self.layer_type == "all"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
        elif (self.layer_type == "scaling"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
        elif (self.layer_type == "bias"):
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
        elif (self.layer_type == "ln"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
            hidden_states = self.delta_vector["activation_ln"](hidden_states)
        if (self.op_position == "res_with_res"):
            hidden_states = hidden_states + x
        return hidden_states


class ActivationLLama(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, base_model, op_position="ffn", layer_type="all", exclude_layers=[]):
        super().__init__()
        self.base_model = base_model
        self.model_type = "llama-7b"
        self.layer_type = layer_type
        self.op_position = op_position
        self.exclude_layers = exclude_layers
        if (exclude_layers):
            pattern_str = '|'.join(map(str, exclude_layers))
            pattern = re.compile(r'\b(?:' + pattern_str + r')\b')
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if (exclude_layers):
                match = pattern.search(key)
                if (match):
                    continue
            if (self.check_update(key)):
                self.replace_layer(key)

                # print(self.print_trainable_parameters())

        # self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    # # if attribute is not in current class, it will look for the attribute in the base_model
    # def __getattr__(self, name):
    #     if name == "prepare_inputs_for_generation":
    #         return getattr(self.base_model, name)
    #     else:
    #         raise AttributeError

    def check_update(self, key):
        if (self.op_position == "ffn"):
            return self.match_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)

    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
            hidden_size=self.base_model.config.hidden_size,
            update_layer=replaced_module,
            layer_type=self.layer_type,
            op_position=self.op_position,
            is_llama=True)
        setattr(parent_module, replaced_name_last, new_module)

    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0
        for name, param in self.base_model.named_parameters():
            total_parameters += param.numel()
            if (param.requires_grad):
                trainable_parameters += param.numel()

        return {
            "total_para:": total_parameters,
            "trainable_para: ": trainable_parameters,
            "trainable%:": f"{100 * trainable_parameters / total_parameters:.4f}"
        }

    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False

    def match_substring(self, input_string):
        pattern = r'down_proj'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                if ("activation_ln" in key):
                    if ("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif ("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)


def get_red_model_state_dict(model):
    state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "delta_vector" in k}


def set_red_model_state_dict(model_red, global_dict, local_dict):
    state_dict = model_red.state_dict()
    for k in global_dict:
        state_dict[k] = global_dict[k]
    for k in local_dict:
        state_dict[k] = local_dict[k]
    model_red.load_state_dict(state_dict, strict=False)
    return model_red


def get_peft_model_state_dict(model):
    state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "lora_" in k}


def set_red_model_params_trainable(model):
    for name, param in model.named_parameters():
        if "delta_vector" in name:
            param.requires_grad = True
    return model


def set_peft_model_params_trainable(model):
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    return model

