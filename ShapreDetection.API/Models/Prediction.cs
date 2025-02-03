using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Tensorboard;

namespace ShapeDetection.API.Models;

/// <summary>
/// Provides methods for predicting shapes in images.
/// </summary>
public class Prediction
{
    private static readonly string ModelPath;
    private static readonly IConfiguration Configuration;

    /// <summary>
    /// Initializes static members of the <see cref="Prediction"/> class.
    /// </summary>
    static Prediction()
    {
        var builder = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);
        Configuration = builder.Build();
        ModelPath = Configuration["modelTraining:modelFile"]!;
    }

    /// <summary>
    /// Lazy-loaded prediction engine for shape detection.
    /// </summary>
    public static readonly Lazy<PredictionEngine<InputModel, OutputModel>> PredictEngine = new Lazy<PredictionEngine<InputModel, OutputModel>>(() => CreatePredictEngine(), true);

    /// <summary>
    /// Predicts the shapes in the given input model.
    /// </summary>
    /// <param name="input">The input model containing the image to analyze.</param>
    /// <returns>The output model containing the prediction results.</returns>
    public static OutputModel Predict(InputModel input)
    {
        return PredictEngine.Value.Predict(input);
    }

    /// <summary>
    /// Creates the prediction engine for shape detection.
    /// </summary>
    /// <returns>A prediction engine for shape detection.</returns>
    private static PredictionEngine<InputModel, OutputModel> CreatePredictEngine()
    {
        var mlContext = new MLContext();
        var model = mlContext.Model.Load(ModelPath, out _);
        return mlContext.Model.CreatePredictionEngine<InputModel, OutputModel>(model);
    }
}
