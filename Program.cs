using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using SkiaSharp;
using System.Collections.Generic;

class Program
{
    static void Main(string[] args)
    {
        using var yolo = new Yolo(new YoloOptions
        {
            OnnxModel = @"yolo11x.onnx",
            ModelType = ModelType.ObjectDetection,
            Cuda = false
        });

        using var image = SKImage.FromEncodedData(@"test.jpg");
        List<ObjectDetection> results = yolo.RunObjectDetection(image, confidence: 0.45, iou: 0.5);

        foreach (var result in results)
        {
            Console.WriteLine($"Label: {result.Label}, Confidence: {result.Confidence}, X: {result.BoundingBox.Left}, Y: {result.BoundingBox.Top}, Width: {result.BoundingBox.Width}, Height: {result.BoundingBox.Height}");
        }

        using var resultImage = image.Draw(results);
        resultImage.Save(@"result.jpg", SKEncodedImageFormat.Jpeg, 80);
    }
}
