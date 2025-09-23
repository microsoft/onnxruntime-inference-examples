/** A downloader which helps to download a serie of files
 *
 * @example
 *
 * ```javascript
 * const downloader = createDownloader({ prefix: 'engle' });
 * const res = await downloader.downloadFile('https://example.com/file.pdf');
 * console.info(res.path());
 * ```
 */

import RNFetchBlob from 'rn-fetch-blob';
import RNFS from 'react-native-fs';

const DOWNLOAD_DIRECTORY = RNFetchBlob.fs.dirs.DocumentDir;

type CreateDownloaderParams = {
  prefix: string;
};

type FetchFunction = ReturnType<typeof RNFetchBlob.config>;

type CreateDownloaderReturn<T extends FetchFunction = FetchFunction> = {
  config: T;
  downloadFile: T['fetch'];
};

/**
 * Creates a downloader instance with the specified prefix.
 *
 * @param {CreateDownloaderParams} options - Options for the downloader.
 * @return {Object} The configured RNFetchBlob instance.
 */
export function createDownloader({
  prefix,
}: CreateDownloaderParams): CreateDownloaderReturn {
  const downloadLocation = `${DOWNLOAD_DIRECTORY}/${prefix}`;
  const rnFetchConfig = RNFetchBlob.config({
    fileCache: true,
    path: downloadLocation,
    addAndroidDownloads: {
      useDownloadManager: true,
      notification: true,
      path: downloadLocation,
      description: 'Downloading a large file...',
      title: downloadLocation,
    },
  });

  return {
    config: rnFetchConfig,
    downloadFile: rnFetchConfig.fetch,
  };
}

type IsFileDownloadedParams = {
  path: string;
  checksum: {
    filesize: number;
  };
};

/**
 * Checks if a downloaded file is safe by verifying its existence and size.
 *
 * @param {IsFileDownloadedParams} params - Options for the file safety check.
 * @param {string} params.path - The path of the file to check.
 * @param {{ filesize: number }} params.checksum - The expected file size.
 * @return {Promise<boolean>} True if the file is safe, false otherwise.
 */
export async function isCompleteDownloaded({
  path,
  checksum,
}: IsFileDownloadedParams): Promise<boolean> {
  const exist = await RNFS.exists(path);
  if (!exist) {
    return false;
  }
  const fileInfo = await RNFS.stat(path);
  return fileInfo.size === checksum.filesize;
}

/**
 * Getting filesize by calling HEAD request to the server
 * @param url
 */
export async function queryFileSize(url: string): Promise<number | null> {
  try {
    const res = await fetch(url, {
      method: 'HEAD',
    });
    const length = res.headers.get('Content-Length');
    return length ? parseInt(length) : null;
  } catch (error) {
    console.warn('Unable to perform HEAD request', error);
    return null;
  }
}
