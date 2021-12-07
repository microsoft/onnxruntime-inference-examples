// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Plugin.Media;
using Plugin.Media.Abstractions;
using Xamarin.Essentials;
using Xamarin.Forms;

namespace VisionSample.Forms
{
    enum ImageAcquisitionMode
    {
        Sample,
        Capture,
        Pick
    }

    public partial class MainPage : ContentPage
    {
        IVisionSample _resnet;
        IVisionSample _ultraface;

        IVisionSample ResNet => _resnet ??= new ResNetSample();
        IVisionSample Ultraface => _ultraface ??= new UltrafaceSample();

        public MainPage()
        {
            InitializeComponent();

            // See:
            // ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/
            // Core ML: https://developer.apple.com/documentation/coreml
            // NNAPI: https://developer.android.com/ndk/guides/neuralnetworks
            ExecutionProviderOptions.Items.Add(nameof(VisionSample.ExecutionProviderOptions.CPU));
            ExecutionProviderOptions.Items.Add(Device.RuntimePlatform == Device.Android ? "NNAPI" : "Core ML");
            ExecutionProviderOptions.SelectedIndex = 0;

            if (ResourceLoader.EmbeddedResourceExists(ResNetSample.ModelFilename))
                Models.Items.Add(ResNet.Name);

            if (ResourceLoader.EmbeddedResourceExists(UltrafaceSample.ModelFilename))
                Models.Items.Add(Ultraface.Name);

            if (Models.Items.Any())
            {
                Models.SelectedIndex = Models.Items.IndexOf(Models.Items.First());
            }
            else
                Models.IsEnabled = false;
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();
            ExecutionProviderOptions.SelectedIndexChanged += ExecutionProviderOptions_SelectedIndexChanged;
            Models.SelectedIndexChanged += Models_SelectedIndexChanged;
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            ExecutionProviderOptions.SelectedIndexChanged -= ExecutionProviderOptions_SelectedIndexChanged;
            Models.SelectedIndexChanged -= Models_SelectedIndexChanged;
        }

        async Task UpdateExecutionProviderAsync() 
        {
            var executionProvider = ExecutionProviderOptions.SelectedItem switch
            {
                nameof(VisionSample.ExecutionProviderOptions.CPU) => VisionSample.ExecutionProviderOptions.CPU,
                _ => VisionSample.ExecutionProviderOptions.Platform
            };

            IVisionSample sample = Models.SelectedItem switch
            {
                ResNetSample.Identifier => ResNet,
                UltrafaceSample.Identifier => Ultraface,
                _ => null
            };

            await sample.UpdateExecutionProviderAsync(executionProvider);
        }

        async Task AcquireAndAnalyzeImageAsync(ImageAcquisitionMode acquisitionMode = ImageAcquisitionMode.Sample)
        {
            byte[] outputImage = null;
            string caption = null;

            try
            {
                SetBusyState(true);

                if (Models.Items.Count == 0 || Models.SelectedItem == null)
                {
                    SetBusyState(false);
                    await DisplayAlert("No Samples", "Model files could not be found", "OK");
                    return;
                }

                var imageData = acquisitionMode switch
                {
                    ImageAcquisitionMode.Capture => await TakePhotoAsync(),
                    ImageAcquisitionMode.Pick => await PickPhotoAsync(),
                    _ => await GetSampleImageAsync()
                };

                if (imageData == null)
                {
                    SetBusyState(false);

                    if (acquisitionMode == ImageAcquisitionMode.Sample)
                        await DisplayAlert("No Sample Image", $"No sample image for {Models.SelectedItem}", "OK");

                    return;
                }

                ClearResult();

                IVisionSample sample = Models.SelectedItem switch
                {
                    ResNetSample.Identifier => ResNet,
                    UltrafaceSample.Identifier => Ultraface,
                    _ => null
                };

                var result = await sample.ProcessImageAsync(imageData);

                outputImage = result.Image;
                caption = result.Caption;
            }
            finally
            {
                SetBusyState(false);
            }

            if (outputImage != null)
                ShowResult(outputImage, caption);
        }

        Task<byte[]> GetSampleImageAsync() => Task.Run(() =>
        {
            var assembly = GetType().Assembly;

            var imageName = Models.SelectedItem switch
            {
                ResNetSample.Identifier => "dog.jpg",
                UltrafaceSample.Identifier => "satya.jpg",
                _ => null
            };

            if (string.IsNullOrWhiteSpace(imageName))
                return null;

            using Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.SampleImages.{imageName}");
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);
            var sampleImage = memoryStream.ToArray();

            return sampleImage;
        });

        async Task<byte[]> PickPhotoAsync()
        {
            FileResult photo;

            try
            {
                photo = await MediaPicker.PickPhotoAsync(new MediaPickerOptions { Title = "Choose photo" });
            }
            catch (FeatureNotSupportedException fnsEx)
            {
                throw new Exception("Feature is not supported on the device", fnsEx);
            }
            catch (PermissionException pEx)
            {
                throw new Exception("Permissions not granted", pEx);
            }
            catch (Exception ex)
            {
                throw new Exception($"The {nameof(PickPhotoAsync)} method threw an exception", ex);
            }

            if (photo == null)
                return null;

            var bytes = await GetBytesFromPhotoFile(photo);

            return SkiaSharpUtils.HandleOrientation(bytes);
        }

        async Task<byte[]> TakePhotoAsync()
        {
            MediaFile photo;

            try
            {
                await CrossMedia.Current.Initialize();

                if (!CrossMedia.Current.IsCameraAvailable || !CrossMedia.Current.IsTakePhotoSupported)
                    throw new Exception("No camera available");

                photo = await CrossMedia.Current.TakePhotoAsync(new StoreCameraMediaOptions()).ConfigureAwait(false);
            }
            catch (FeatureNotSupportedException fnsEx)
            {
                throw new Exception("Feature is not supported on the device", fnsEx);
            }
            catch (PermissionException pEx)
            {
                throw new Exception("Permissions not granted", pEx);
            }
            catch (Exception ex)
            {
                throw new Exception($"The {nameof(TakePhotoAsync)} method throw an exception", ex);
            }

            if (photo == null)
                return null;

            var bytes = await GetBytesFromPhotoFile(photo);
            photo.Dispose();

            return SkiaSharpUtils.HandleOrientation(bytes);
        }

        async Task<byte[]> GetBytesFromPhotoFile(MediaFile fileResult)
        {
            byte[] bytes;

            using Stream stream = await Task.Run(() => fileResult.GetStream());
            using MemoryStream ms = new MemoryStream();

            stream.CopyTo(ms);
            bytes = ms.ToArray();

            return bytes;
        }

        async Task<byte[]> GetBytesFromPhotoFile(FileResult fileResult)
        {
            byte[] bytes;

            using Stream stream = await fileResult.OpenReadAsync();
            using MemoryStream ms = new MemoryStream();

            stream.CopyTo(ms);
            bytes = ms.ToArray();

            return bytes;
        }

        void ClearResult() => MainThread.BeginInvokeOnMainThread(() =>
        {
            OutputImage.Source = null;
            Caption.Text = string.Empty;
        });

        void ShowResult(byte[] image, string caption = null) => MainThread.BeginInvokeOnMainThread(() =>
        {
            OutputImage.Source = ImageSource.FromStream(() => new MemoryStream(image));
            Caption.Text = string.IsNullOrWhiteSpace(caption) ? string.Empty : caption;
        });

        void SetBusyState(bool busy)
        {
            ExecutionProviderOptions.IsEnabled = !busy;
            SamplePhotoButton.IsEnabled = !busy;
            PickPhotoButton.IsEnabled = !busy;
            TakePhotoButton.IsEnabled = !busy;
            BusyIndicator.IsEnabled = busy;
            BusyIndicator.IsRunning = busy;
        }

        ImageAcquisitionMode GetAcquisitionModeFromText(string tag) => tag switch
        {
            nameof(ImageAcquisitionMode.Capture) => ImageAcquisitionMode.Capture,
            nameof(ImageAcquisitionMode.Pick) => ImageAcquisitionMode.Pick,
            _ => ImageAcquisitionMode.Sample
        };

        void AcquireButton_Clicked(object sender, EventArgs e)
            => AcquireAndAnalyzeImageAsync(GetAcquisitionModeFromText((sender as Button).Text)).ContinueWith((task)
                => { if (task.IsFaulted) MainThread.BeginInvokeOnMainThread(()
                  => DisplayAlert("Error", task.Exception.Message, "OK")); });

        private void ExecutionProviderOptions_SelectedIndexChanged(object sender, EventArgs e)
            => UpdateExecutionProviderAsync().ContinueWith((task)
                => { if (task.IsFaulted) MainThread.BeginInvokeOnMainThread(() 
                    => DisplayAlert("Error", task.Exception.Message, "OK"));});

        private void Models_SelectedIndexChanged(object sender, EventArgs e)
            // make sure EP is in sync
            => ExecutionProviderOptions_SelectedIndexChanged(null, null);
    }
}
