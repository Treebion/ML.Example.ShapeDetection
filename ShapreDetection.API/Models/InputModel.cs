using Microsoft.ML.Data;

namespace ShapeDetection.API.Models;

/// <summary>
/// Represents the input model for shape prediction.
/// </summary>
public class InputModel
{
    /// <summary>
    /// Gets or sets the labels associated with the image.
    /// </summary>
    [LoadColumn(0)]
    [ColumnName(@"Labels")]
    public string[] Labels { get; set; }

    /// <summary>
    /// Gets or sets the image to be analyzed.
    /// </summary>
    [LoadColumn(1)]
    [ColumnName(@"Image")]
    [Microsoft.ML.Transforms.Image.ImageType(800, 600)]
    public MLImage Image { get; set; }

    /// <summary>
    /// Gets or sets the bounding box coordinates for the detected shapes.
    /// </summary>
    [LoadColumn(2)]
    [ColumnName(@"Box")]
    public float[] Box { get; set; }
}
