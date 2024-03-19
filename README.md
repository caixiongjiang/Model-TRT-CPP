# Model-TRT-CPP

## Result
<table>
    <tr>
	    <th colspan="6">模型推理效果对比</th>
	</tr >
	<tr>
	  <td  style="text-align: center;"><b>状态</b></td>
      <td  style="text-align: center;"><b>任务</b></td>
	  <td  style="text-align: center;"><b>模型</b></td>
      <td  style="text-align: center;"><b>分辨率</b></td> 
      <td  style="text-align: center;"><b>Pytorch推理(FPS)</b></td>
      <td  style="text-align: center;"><b>TensorRT推理(FPS)</b></td>
	</tr >
    <tr> 
      <td style="text-align: center;"><input type="checkbox" checked></td> 
      <td rowspan="4" style="text-align: center;">分类</td>
      <td style="text-align: center;"><span style="display: block; text-align: center;">ResNet50</td>
      <td style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr> 
      <td style="text-align: center;"><input type="checkbox" ></td> 
      <td style="text-align: center;"><span style="display: block; text-align: center;">MobileNet</td>
      <td style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr> 
      <td style="text-align: center;"><input type="checkbox" ></td> 
      <td style="text-align: center;"><span style="display: block; text-align: center;">Vision Transformer</td>
      <td style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr> 
      <td style="text-align: center;"><input type="checkbox" ></td> 
      <td style="text-align: center;">Swin Transformer</td>
      <td style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
</table>