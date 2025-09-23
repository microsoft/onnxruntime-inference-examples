import { useState } from 'react';
import { StyleSheet, View, Text, Button } from 'react-native';
import { ask, init } from 'react-native-phi3';
import RNFetchBlob from 'rn-fetch-blob';
import { ASSETS, type Asset } from './downloader/phi3';
import {
  createDownloader,
  isCompleteDownloaded,
  queryFileSize,
} from './downloader';

const MODEL_DIRECTORY = RNFetchBlob.fs.dirs.DocumentDir + '/engle/phi3';

export default function App() {
  
  const [ready, setReady] = useState<boolean>(false);
  const [curAsset, setCurAsset] = useState<Asset | undefined | null>(undefined);
  const [phi3Result, setPhi3Result] = useState<string | undefined>();
  const [phi3Loaded, setPhi3Loaded] = useState<boolean>(false);

  return (
    <View style={styles.container}>
      <Button
        disabled={ready || curAsset === null || curAsset}
        onPress={async () => {
          for (const asset of ASSETS) {
            setCurAsset(asset);
            const path = RNFetchBlob.fs.dirs.DocumentDir + '/' + asset.prefix;
            const filesize = await queryFileSize(asset.url);
            const shouldDownload = filesize
              ? !(await isCompleteDownloaded({
                  path,
                  checksum: {
                    filesize: filesize,
                  },
                }))
              : true;
            if (shouldDownload) {
              console.info('downloading ', asset.prefix);
              const downloader = createDownloader({
                prefix: asset.prefix,
              });
              await downloader.downloadFile('GET', asset.url);
            }
            setCurAsset(null);
          }
          setReady(true);
        }}
        title={'Download Asset'}
      />
      <Text>
        Assets State:{' '}
        {ready ? 'Yes' : curAsset === null ? 'Downloading' : 'Not ready'}
      </Text>
      <Text>Phi3 Loaded: {phi3Loaded ? 'Yes' : 'No'}</Text>
      <Text>Phi3 Response: {phi3Result}</Text>
      <Button
        onPress={() => {
          console.info(MODEL_DIRECTORY)
          init(MODEL_DIRECTORY)
            .then((success: boolean) => {
              setPhi3Loaded(success);
            })
            .catch((error: any) => {
              console.info(
                `Error while init the model, Please make sure you path points to the correct model path.
You can download the model here: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
`
              );
              console.info(error.message);
              // console.info(error.stack)
              console.info(error.userInfo);
            });
        }}
        title="Start Loading model"
      />
      <Button
        disabled={!phi3Loaded}
        onPress={() => {
          console.info('doing something here');
          ask('hello')
            .then(setPhi3Result)
            .catch((error) => {
              console.info('unable to inference information ', error);
            });
        }}
        title="Running text for Phi3"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  box: {
    width: 60,
    height: 60,
    marginVertical: 20,
  },
});
