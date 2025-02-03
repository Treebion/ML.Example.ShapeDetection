using Microsoft.ML.Data;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Net_Shape_Detection;

internal class ModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"Labels")]
    public string[] Labels { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"Image")]
    [Microsoft.ML.Transforms.Image.ImageType(256, 256)]
    public MLImage Image { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"Box")]
    public float[] Box { get; set; }
}
