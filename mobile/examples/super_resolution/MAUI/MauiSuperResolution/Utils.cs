using Microsoft.ML.OnnxRuntime;

namespace MauiSuperResolution
{
    internal class Utils
    {
        internal static async Task<byte[]> LoadResource(string name)
        {
            using Stream fileStream = await FileSystem.Current.OpenAppPackageFileAsync(name);
            using MemoryStream memoryStream = new MemoryStream();
            fileStream.CopyTo(memoryStream);
            return memoryStream.ToArray();
        }

        internal static Task<byte[]> GetSampleImageAsync() => Task.Run(
            () =>        
            { 
                return LoadResource("lr_lion.png").Result;
            });

        internal static async Task<byte[]> PickPhotoAsync()
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
            {
                return null;
            }

            return await GetBytesFromPhotoFile(photo);
        }

        internal static async Task<byte[]> TakePhotoAsync()
        {
            FileResult photo;

            try
            {
                if (!MediaPicker.Default.IsCaptureSupported)
                {
                    throw new FeatureNotSupportedException("Image capture is not supported.");
                }

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
                throw new Exception("Capturing a photo with MAUI MediaPicker does not work on Windows currently.");
#else
#if IOS
                if (Microsoft.Maui.Devices.DeviceInfo.Current.DeviceType == Microsoft.Maui.Devices.DeviceType.Virtual)
                {
                    // https://github.com/dotnet/maui/issues/7013#issuecomment-1123384958
                    throw new Exception("Capturing a photo with MAUI MediaPicker is not possible with the iOS simulator.");
                }
#endif
                // if it wasn't one of the above special cases the user most likely chose not to capture an image
                return null;
#endif
            }

            return await GetBytesFromPhotoFile(photo);
        }

        internal static async Task<byte[]> GetBytesFromPhotoFile(FileResult fileResult)
        {
            using Stream stream = await fileResult.OpenReadAsync();
            using MemoryStream ms = new MemoryStream();

            stream.CopyTo(ms);
            return ms.ToArray();
        }

        internal static async Task<byte[]> GetInputImageAsync(ImageAcquisitionMode acquisitionMode)
        {
            var imageData = acquisitionMode switch
            {
                ImageAcquisitionMode.Capture => await TakePhotoAsync(),
                ImageAcquisitionMode.Pick => await PickPhotoAsync(),
                _ => await GetSampleImageAsync()
            };

            return imageData;
        }
    }
}
