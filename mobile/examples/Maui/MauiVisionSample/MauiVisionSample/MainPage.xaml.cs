// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace MauiVisionSample;

//using Microsoft.Maui.Platform;
//using Microsoft.ML.OnnxRuntime;

enum ImageAcquisitionMode
{
    Sample,
    Capture,
    Pick
}

public partial class MainPage : ContentPage
{
    IVisionSample _mobilenet;
    IVisionSample _ultraface;

    IVisionSample Mobilenet => _mobilenet ??= new MobilenetSample();
    IVisionSample Ultraface => _ultraface ??= new UltrafaceSample();

    public MainPage()
	{
		InitializeComponent();

        // See:
        // ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/
        // Core ML: https://developer.apple.com/documentation/coreml
        // NNAPI: https://developer.android.com/ndk/guides/neuralnetworks
        ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.CPU));

        if (DeviceInfo.Platform == DevicePlatform.Android)
        {
            ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.NNAPI));
        }

        if (DeviceInfo.Platform == DevicePlatform.iOS)
        {
            ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.CoreML));
        }

        ExecutionProviderOptions.SelectedIndex = 0;

        if (FileSystem.Current.AppPackageFileExistsAsync(MobilenetSample.ModelFilename).Result)
        {
            Models.Items.Add(Mobilenet.Name);
        }

        if (FileSystem.Current.AppPackageFileExistsAsync(UltrafaceSample.ModelFilename).Result)
        {
            Models.Items.Add(Ultraface.Name);
        }

        if (Models.Items.Any())
        {
            Models.SelectedIndex = Models.Items.IndexOf(Models.Items.First());
        }
        else
        { 
            Models.IsEnabled = false; 
        }
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
            nameof(ExecutionProviders.CPU)    => ExecutionProviders.CPU,
            nameof(ExecutionProviders.NNAPI)  => ExecutionProviders.NNAPI,
            nameof(ExecutionProviders.CoreML) => ExecutionProviders.CoreML,
            _ => ExecutionProviders.CPU
        };

        IVisionSample sample = Models.SelectedItem switch
        {
            MobilenetSample.Identifier => Mobilenet,
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
                MobilenetSample.Identifier => Mobilenet,
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
        {
            ShowResult(outputImage, caption);
        }
    }

    Task<byte[]> GetSampleImageAsync() => Task.Run(() =>
    {
        var assembly = GetType().Assembly;

        var imageName = Models.SelectedItem switch
        {
            MobilenetSample.Identifier => "wolves.jpg",
            UltrafaceSample.Identifier => "satya.jpg",
            _ => null
        };

        if (string.IsNullOrWhiteSpace(imageName))
        {
            return null;
        }

        return Utils.LoadResource(imageName).Result;
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

        return Utils.HandleOrientation(bytes);
    }

    async Task<byte[]> TakePhotoAsync()
    {
        if (!MediaPicker.Default.IsCaptureSupported)
        {
            return null;
        }

        FileResult photo;

        try
        {
            photo = await MediaPicker.Default.CapturePhotoAsync();
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
        {
#if WINDOWS
            // MediaPicker CapturePhotoAsync does not work on Windows currently.
            // https://github.com/dotnet/maui/issues/7616 eventually links to 
            // https://github.com/microsoft/WindowsAppSDK/issues/1034 which is apparently the cause. 
            // https://github.com/dotnet/maui/issues/7660#issuecomment-1152347557 has an example alternative 
            // implementation that could be used instead of MediaPicker.CapturePhotoAsync on Windows, although ideally
            // MAUI would do that internally...
            await DisplayAlert("Sorry", "Capturing a photo with MAUI MediaPicker does not work on Windows currently.", "OK");
#elif IOS
            if (Microsoft.Maui.Devices.DeviceInfo.Current.DeviceType == Microsoft.Maui.Devices.DeviceType.Virtual)
            {
                // https://github.com/dotnet/maui/issues/7013#issuecomment-1123384958
                await DisplayAlert("Sorry", "Capturing a photo with MAUI MediaPicker is not possible with the iOS simulator.", "OK");
            }
#endif
            // if it wasn't one of the above special cases the user most likely chose not to capture an image
            return null;
        }

        var bytes = await GetBytesFromPhotoFile(photo);

        return Utils.HandleOrientation(bytes);
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
            => {
                if (task.IsFaulted) MainThread.BeginInvokeOnMainThread(()
              => DisplayAlert("Error", task.Exception.Message, "OK"));
            });

    private void ExecutionProviderOptions_SelectedIndexChanged(object sender, EventArgs e)
        => UpdateExecutionProviderAsync().ContinueWith((task)
            => {
                if (task.IsFaulted) MainThread.BeginInvokeOnMainThread(()
                => DisplayAlert("Error", task.Exception.Message, "OK"));
            });

    private void Models_SelectedIndexChanged(object sender, EventArgs e)
        // make sure EP is in sync
        => ExecutionProviderOptions_SelectedIndexChanged(null, null);
}


