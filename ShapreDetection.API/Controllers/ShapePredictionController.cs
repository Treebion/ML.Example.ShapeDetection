using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using ShapeDetection.API.Models;
using System.Drawing;
using System.Drawing.Imaging;

namespace ShapeDetection.API.Controllers
{
    /// <summary>
    /// Controller for handling shape prediction requests.
    /// </summary>
    [ApiController]
    [Route("[controller]")]
    public class ShapePredictionController : ControllerBase
    {
        private readonly Prediction _prediction;

        /// <summary>
        /// Initializes a new instance of the <see cref="ShapePredictionController"/> class.
        /// </summary>
        /// <param name="prediction">The prediction service.</param>
        public ShapePredictionController(Prediction prediction)
        {
            _prediction = prediction;
        }

        /// <summary>
        /// Predicts the shape in the uploaded image.
        /// </summary>
        /// <param name="image">The image file to analyze.</param>
        /// <returns>An <see cref="IActionResult"/> containing the prediction result.</returns>
        [HttpPost("predict")]
        public async Task<IActionResult> PredictShape(IFormFile image)
        {
            if (image == null || image.Length == 0)
            {
                return BadRequest("No image uploaded.");
            }

            using var stream = new MemoryStream();
            await image.CopyToAsync(stream);
            stream.Position = 0;
            var inputModel = new InputModel
            {
                Image = MLImage.CreateFromStream(stream)
            };

            if (inputModel.Image.Height != 256 || inputModel.Image.Width != 256)
            {
                return BadRequest("Image must be 256 x 256");
            }

            var result = Prediction.Predict(inputModel);

            // Create a new MemoryStream for the Bitmap
            using var imageStream = new MemoryStream(stream.ToArray());
            using var bitmap = new Bitmap(imageStream);
            using var graphics = Graphics.FromImage(bitmap);
            var pen = new Pen(Color.Red, 2);
            var font = new Font("Arial", 12);
            var brush = new SolidBrush(Color.Yellow);

            for (int i = 0; i < result.PredictedBoundingBoxes.Length; i += 4)
            {
                var x = result.PredictedBoundingBoxes[i];
                var y = result.PredictedBoundingBoxes[i + 1];
                var width = result.PredictedBoundingBoxes[i + 2] - result.PredictedBoundingBoxes[i];
                var height = result.PredictedBoundingBoxes[i + 3] - result.PredictedBoundingBoxes[i + 1];

                graphics.DrawRectangle(pen, x, y, width, height);
                graphics.DrawString(result.PredictedLabel[i / 4], font, brush, x, y - 20);
            }

            using var outputStream = new MemoryStream();
            bitmap.Save(outputStream, ImageFormat.Png);
            outputStream.Position = 0;

            // Copy the output stream to a new MemoryStream to avoid disposing issues
            var finalStream = new MemoryStream(outputStream.ToArray());

            return File(finalStream, "image/png");
        }
    }
}
