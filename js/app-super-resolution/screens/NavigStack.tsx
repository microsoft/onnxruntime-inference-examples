import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { NavigationContainer } from '@react-navigation/native';
import { StackScreenProps } from '@react-navigation/stack';
import screens from "./screens"


export type RootParamList = {
  WelcomeScreen: undefined
  MainScreen: undefined
}

export type MainScreenProps = StackScreenProps<RootParamList, "MainScreen", "1">


const Root = createNativeStackNavigator<RootParamList>();

export default function App() {
  return (
    <NavigationContainer>
      <Root.Navigator
        initialRouteName='WelcomeScreen'
        screenOptions={{
            headerTintColor: 'white',
            headerStyle: { backgroundColor: '#118ab2' },
        }}>
        <Root.Screen name='WelcomeScreen' component={screens.first}/>
        <Root.Screen name='MainScreen' component={screens.second}/>
      </Root.Navigator>
    </NavigationContainer>
  )
}

