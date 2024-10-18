using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using SkiaSharp;

class Program
{
    static void Main(string[] args)
    {
        // Instantiate a new Yolo object
        using var yolo = new Yolo(new YoloOptions
        {
            OnnxModel = @"yolo11x.onnx",  
            ModelType = ModelType.ObjectDetection,
            Cuda = false  
        });

        using var image = SKImage.FromEncodedData(@"test.jpg");

        var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

        using var resultImage = image.Draw(results);

        resultImage.Save(@"result.jpg", SKEncodedImageFormat.Jpeg, 80);
    }
}
