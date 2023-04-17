import { configureStore } from '@reduxjs/toolkit'
import mlPredictionsReducer from './similarPred'
import userStore from './user'

const store = configureStore({
    reducer: {
        mlPrediction: mlPredictionsReducer,
        userStore: userStore
    }
})

export default store