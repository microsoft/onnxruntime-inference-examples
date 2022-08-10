import { StyleSheet } from "react-native";


export const styles = StyleSheet.create({
  buttonContainer: {
    backgroundColor: 'white'
  },

  cameraButton: {
    alignSelf: "flex-start",
    margin: 8,
    width: 40,
    height: 40,
    resizeMode: "contain"
  },

  containerAndroid: {
      flex: 1,
      backgroundColor: '#ffb703',
      alignItems: 'stretch',
      justifyContent: 'center',
  },

  containerWeb: {
    flex: 1,
    backgroundColor: '#ffb703',
    alignItems: 'center',
    justifyContent: 'center',
},

  imageView: {
    flexDirection: "row", 
    padding: 20, 
    justifyContent:"center", 
    alignItems: "center",
  },

  instructions: {
      color: 'white',
      marginBottom: 10,
      alignSelf: "center"
  },

  item: {
      margin: 24,
      fontSize: 18,
      fontWeight: "bold",
      textAlign: "center"
  },
    
  scrollView: {
    backgroundColor: "#e9c46a",
    padding: 4,
  },

  thumbnail: {
    alignSelf: "center",
    margin: 8,
    width: 350,
    height: 350,
    resizeMode: "contain"
  },

  userInput: {
    flexDirection: "row",
    justifyContent: "space-around",
    padding: 8,
    marginLeft: 10,
    marginRight: 10,
    alignItems: "center",
  },

  welcomeText: {
    marginBottom: 16, 
    margin: 24, 
    fontSize: 18, 
    fontWeight: "bold", 
    textAlign: "center"
  }
  });
