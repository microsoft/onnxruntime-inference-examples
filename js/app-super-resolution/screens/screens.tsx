import WelcomeScreen from './welcomeScreen';
import AndroidApp from './mobile';
import WebApp from './web';
import { Platform } from 'react-native';


const screens = {
    first : WelcomeScreen,
    second : AndroidApp,
}

if (Platform.OS == "web") {
    screens.second = WebApp
}


export default screens
  