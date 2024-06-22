from typing import Union, Optional, Sequence, Dict, Any 
 
import torch 
import tensorrt as trt 
 
# TODO:支持YOLOv5和v8的TRT推理适配

class TRTWrapper(torch.nn.Module): 
    def __init__(self, engine: Union[str, trt.ICudaEngine], 
                 output_names: Optional[Sequence[str]] = None) -> None: 
        super().__init__() 
        self.engine = engine 
        if isinstance(self.engine, str): 
            # 初始化TensorRT引擎并读取trt模型推理文件
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: 
                with open(self.engine, mode='rb') as f: 
                    engine_bytes = f.read() 
                self.engine = runtime.deserialize_cuda_engine(engine_bytes) 
        # 创建推理上下文
        self.context = self.engine.create_execution_context() 
        names = [_ for _ in self.engine] 
        input_names = list(filter(self.engine.binding_is_input, names)) 
        self._input_names = input_names 
        self._output_names = output_names 
        self.trt_name = engine.split("/")[-1]
 
        if self._output_names is None: 
            output_names = list(set(names) - set(input_names)) 
            self._output_names = output_names 

    def forward(self, inputs: Dict[str, torch.Tensor]): 
        assert self._input_names is not None 
        assert self._output_names is not None 
        bindings = [None] * (len(self._input_names) + len(self._output_names)) 
        profile_id = 0 
        for input_name, input_tensor in inputs.items(): 
            # 检查输入是否有效 一个engine规定了其特殊的输入尺寸范围
            profile = self.engine.get_profile_shape(profile_id, input_name) 
            assert input_tensor.dim() == len( 
                profile[0]), 'Input dim is different from engine profile.' 
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, 
                                             profile[2]): 
                assert s_min <= s_input <= s_max, 'Input shape should be between ' + \
                    f'{profile[0]} and {profile[2]}' + \
                    f' but get {tuple(input_tensor.shape)}.' 
            idx = self.engine.get_binding_index(input_name) 
 
            # 所有输入必须位于GPU上 外层限定单卡
            assert 'cuda' in input_tensor.device.type 
            input_tensor = input_tensor.contiguous() 
            if input_tensor.dtype == torch.long: 
                input_tensor = input_tensor.int() 
            self.context.set_binding_shape(idx, tuple(input_tensor.shape)) 
            bindings[idx] = input_tensor.contiguous().data_ptr() 
 
        # create output tensors 
        outputs = {} 
        for output_name in self._output_names: 
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32 
            shape = tuple(self.context.get_binding_shape(idx)) 
 
            device = torch.device('cuda') 
            output = torch.empty(size=shape, dtype=dtype, device=device) 
            outputs[output_name] = output 
            bindings[idx] = output.data_ptr() 
        self.context.execute_async_v2(bindings, 
                                      torch.cuda.current_stream().cuda_stream) 
        return outputs 