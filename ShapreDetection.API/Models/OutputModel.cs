using Microsoft.ML.Data;

namespace ShapeDetection.API.Models;

/// <summary>
/// Represents the output model for shape prediction.
/// </summary>
public class OutputModel
{
    /// <summary>
    /// Gets or sets the labels associated with the detected shapes.
    /// </summary>
    [ColumnName(@"Labels")]
    public uint[] Labels { get; set; }

    /// <summary>
    /// Gets or sets the image that was analyzed.
    /// </summary>
    [ColumnName(@"Image")]
    [Microsoft.ML.Transforms.Image.ImageType(256, 256)]
    public MLImage Image { get; set; }

    /// <summary>
    /// Gets or sets the bounding box coordinates for the detected shapes.
    /// </summary>
    [ColumnName(@"Box")]
    public float[] Box { get; set; }

    /// <summary>
    /// Gets or sets the predicted labels for the detected shapes.
    /// </summary>
    [ColumnName(@"PredictedLabel")]
    public string[] PredictedLabel { get; set; }

    /// <summary>
    /// Gets or sets the confidence scores for the detected shapes.
    /// </summary>
    [ColumnName(@"Score")]
    public float[] Score { get; set; }

    /// <summary>
    /// Gets or sets the predicted bounding box coordinates for the detected shapes.
    /// </summary>
    [ColumnName(@"PredictedBoundingBoxes")]
    public float[] PredictedBoundingBoxes { get; set; }
}
