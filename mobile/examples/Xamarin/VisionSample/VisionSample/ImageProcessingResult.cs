namespace VisionSample
{
    public class ImageProcessingResult
    {
        public byte[] Image { get; private set; }
        public string Caption { get; private set; }

        internal ImageProcessingResult(byte[] image, string caption = null)
        {
            Image = image;
            Caption = caption;
        }
    }
}