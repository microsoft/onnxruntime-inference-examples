namespace MauiSuperResolution;

using System;

enum ImageAcquisitionMode
{
    Sample,
    Capture,
    Pick
}

public partial class MainPage : ContentPage
{
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

        // XNNPACK provides optimized CPU execution on ARM64 and ARM platforms for models using float
        var arch = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture;
        if (arch == System.Runtime.InteropServices.Architecture.Arm64 ||
            arch == System.Runtime.InteropServices.Architecture.Arm)
        {
            ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.XNNPACK));
        }

        ExecutionProviderOptions.SelectedIndex = 0; // default to CPU
        ExecutionProviderOptions.SelectedIndexChanged += ExecutionProviderOptions_SelectedIndexChanged;

        _currentExecutionProvider = ExecutionProviders.CPU;

        // start creating session in background
        _inferenceSessionCreationTask = CreateInferenceSession();
    }

    private void SampleButton_Clicked(object sender, EventArgs e)
    {
        Run(ImageAcquisitionMode.Sample);
    }

    private void PickButton_Clicked(object sender, EventArgs e)
    {
        Run(ImageAcquisitionMode.Pick);
    }

    private void CaptureButton_Clicked(object sender, EventArgs e)
    {
        Run(ImageAcquisitionMode.Capture);
    }

    // helper to call the async Run with a failure handler
    private void Run(ImageAcquisitionMode mode)
    {
        RunAsync(mode).ContinueWith(resultTask =>
                                    {
                                        if (resultTask.IsFaulted)
                                        {
                                            MainThread.BeginInvokeOnMainThread(
                                                () => DisplayAlert("Error", resultTask.Exception.Message, "OK"));
                                        }
                                    });
    }

    private async Task RunAsync(ImageAcquisitionMode mode)
    {
        await ClearResult();

        byte[] imageBytes = await Utils.GetInputImageAsync(mode);

        if (imageBytes != null)
        {
            await MainThread.InvokeOnMainThreadAsync(
                () => { AfterCaption.Text = "Running inference... please be patient"; });

            await SetBusy(true);

            // create inference session if it doesn't exist or EP has changed
            await CreateInferenceSession();

            // this is an expensive model so execution time can be quite long.
            var outputImageBytes = await Task.Run<byte[]>(
                () => { return _inferenceSession.Run(imageBytes); });

            await SetBusy(false);

            ShowResult(imageBytes, outputImageBytes, _inferenceSession.LastRunTimeMs);
        };
    }

    private async Task CreateInferenceSession()
    {
        // wait if we're already creating an inference session.
        if (_inferenceSessionCreationTask != null)
        {
            await _inferenceSessionCreationTask.ConfigureAwait(false);
            _inferenceSessionCreationTask = null;
        }

        var executionProvider = ExecutionProviderOptions.SelectedItem switch {
            nameof(ExecutionProviders.NNAPI) => ExecutionProviders.NNAPI,
            nameof(ExecutionProviders.CoreML) => ExecutionProviders.CoreML,
            nameof(ExecutionProviders.XNNPACK) => ExecutionProviders.XNNPACK,
            _ => ExecutionProviders.CPU
        };

        if (_inferenceSession == null || executionProvider != _currentExecutionProvider)
        {
            _currentExecutionProvider = executionProvider;

            // re/create an inference session with the execution provider.
            // this is an expensive operation as we have to reload the model, and should be avoided in production apps.
            // recommendation: run the model as a background task with example input for each possible execution
            // provider on the current platform and choose the one with the best performance. this is a one-time
            // operation.
            _inferenceSession = new OrtInferenceSession(_currentExecutionProvider);
            await _inferenceSession.Create();
        }
    }

    private void ExecutionProviderOptions_SelectedIndexChanged(object sender, EventArgs e)
    {
        ExecutionProviderOptions.IsEnabled = false; // disable until session is created
        _inferenceSessionCreationTask = CreateInferenceSession();
        _inferenceSessionCreationTask.ContinueWith(
            (task) =>
            {
                MainThread.BeginInvokeOnMainThread(
                    () =>
                    {
                        ExecutionProviderOptions.IsEnabled = true;
                        if (task.IsFaulted)
                        {
                            DisplayAlert("Error", task.Exception.Message, "OK");
                        }
                    });
            });
    }

    private async Task SetBusy(bool busy)
        => await MainThread.InvokeOnMainThreadAsync(
            () =>
            {
                BusyIndicator.IsRunning = busy;
                BusyIndicator.IsVisible = busy;
            });

    private async Task ClearResult() 
        => await MainThread.InvokeOnMainThreadAsync(
            () =>
            {
                BeforeImage.Aspect = OperatingSystem.IsWindows() ? Aspect.Center : Aspect.AspectFit;
                BeforeImage.Source = "blank.png";
                AfterImage.Source = "onnxruntime_logo.png";
            });

    private void ShowResult(byte[] beforeBytes, byte[] afterBytes, long runMs) 
        => MainThread.BeginInvokeOnMainThread(
            () =>
            {
                BeforeImage.Aspect = OperatingSystem.IsWindows() ? Aspect.Center : Aspect.AspectFit;
                BeforeImage.Source = ImageSource.FromStream(() => new MemoryStream(beforeBytes));

                AfterCaption.Text = "Super Resolution Result (Run took " + runMs + "ms)";
                AfterImage.Source = ImageSource.FromStream(() => new MemoryStream(afterBytes));
            });

    private ExecutionProviders _currentExecutionProvider;
    private OrtInferenceSession _inferenceSession;
    private Task _inferenceSessionCreationTask;
}
