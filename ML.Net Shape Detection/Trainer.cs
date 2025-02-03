using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.AutoFormerV2;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

namespace ML.Net_Shape_Detection;

internal class Trainer
{
    private const int trainingImageWidth = 256;
    private const int trainingImageHeight = 256;

    private readonly IConfiguration _configuration;

    private readonly string trainingFolder;
    private readonly string cocoFilePath;
    private readonly string outputFile;
    private readonly bool useGpu;

    public Trainer(IConfiguration configuration)
    {
        _configuration = configuration;
        trainingFolder = _configuration["modelTraining:imgFolder"]!;
        cocoFilePath = _configuration["modelTraining:cocoFile"]!;
        outputFile = _configuration["modelTraining:modelFile"]!;
        useGpu = bool.Parse(_configuration["modelTraining:useGpu"]!);
    }

    public void Go()
    {
        var mlContext = new MLContext();
        mlContext.GpuDeviceId = useGpu ? 0 : null!;
        mlContext.FallbackToCpu = !useGpu;

        var dataView = DataViewFromCoco(mlContext, cocoFilePath);
        var model = TrainModel(mlContext, dataView);
        SaveModel(mlContext, model, dataView, outputFile);
    }

    public static IDataView DataViewFromCoco(MLContext mLContext, string cocoFilePath) => mLContext.Data.LoadFromEnumerable(LoadCoco(cocoFilePath));

    private static IEnumerable<ModelInput> LoadCoco(string cocoFilePath)
    {
        JsonNode jsonNode;
        using (var sr = new StreamReader(cocoFilePath))
        {
            string json = sr.ReadToEnd();
            jsonNode = JsonSerializer.Deserialize<JsonNode>(json)!;
        };

        var imgData = new List<ModelInput>();
        var categories = jsonNode["categories"]!.AsArray().ToDictionary(k => k!["id"]!.GetValue<int>(), v => v!["name"]!.GetValue<string>());
        var images = jsonNode["images"]!.AsArray().ToDictionary(
            k => k!["id"]!.GetValue<int>(),
            v =>
            {
                var imgPath = v["file_name"]!.GetValue<string>();
                var imgWidth = v["width"]!.GetValue<int>();
                var imgHeight = v["height"]!.GetValue<int>();

                return new { ImgPath = imgPath, ImgWidth = imgWidth, ImgHeight = imgHeight };
            });

        var annotations = jsonNode["annotations"]!.AsArray().Select(s =>
        {
            var imageId = s!["image_id"]!.GetValue<int>();
            var catId = s!["category_id"]!.GetValue<int>();
            var boundingBox = s!["bbox"]!.AsArray().Select(s => s!.GetValue<float>()).ToArray();
            var isCrowd = s!["iscrowd"]!.GetValue<int>();

            return new { ImageId = imageId, CatId = catId, BoundingBox = boundingBox, IsCrowd = isCrowd };
        });

        foreach (var group in annotations.GroupBy(g => g.ImageId))
        {
            var img = images[group.Key];
            var width = img.ImgWidth;
            var labels = group.Select(s => categories[s.CatId]).ToArray();
            var boxes = group.Select(s => new[] {
                s.BoundingBox[0],
                s.BoundingBox[1],
                s.BoundingBox[0] + s.BoundingBox[2],
                s.BoundingBox[1] + s.BoundingBox[3],
            }).SelectMany(sm => sm).ToArray();

            var mlImg = MLImage.CreateFromFile(img.ImgPath);
            imgData.Add(new ModelInput
            {
                Image = mlImg,
                Labels = labels,
                Box = boxes
            });
        }

        return imgData;
    }

    private static ITransformer TrainModel(MLContext mLContext, IDataView dataView)
    {
        var pipeline = CreatePipeLine(mLContext);
        var model = pipeline.Fit(dataView);

        return model;
    }

    private static IEstimator<ITransformer> CreatePipeLine(MLContext mLContext)
    {
        return mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Labels", inputColumnName: "Labels", addKeyValueAnnotationsAsText: false)
            .Append(mLContext.MulticlassClassification.Trainers.ObjectDetection(new ObjectDetectionTrainer.Options()
            {
                LabelColumnName = "Labels",
                PredictedLabelColumnName = "PredictedLabel",
                BoundingBoxColumnName = "Box",
                ImageColumnName = "Image",
                ScoreColumnName = "Score",
                MaxEpoch = 5,
                InitLearningRate = 1,
                WeightDecay = 0,
                LogEveryNStep = 10
            }))
            .Append(mLContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));
    }

    public static void SaveModel(MLContext mlContext, ITransformer model, IDataView dataView, string modelSavePath)
    {
        using (var fs = File.Create(modelSavePath))
        {
            mlContext.Model.Save(model, dataView.Schema, fs);
        }
    }

}
