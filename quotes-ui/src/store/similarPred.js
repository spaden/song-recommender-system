import { createSlice } from '@reduxjs/toolkit'
import service from '../service/service'

const initialState = {
    predictions: '',
    showPredictedModal: false,
    userInput: '',
    status: {
        isFetching: false,
        isFetched: false,
        error: false
    }
}

const mlPredictions = createSlice({
    name: 'mlPredictions',
    initialState: initialState,
    reducers: {
        FETCH_PREDICTIONS_REQUEST(state) {
            const { status } = state
            status.isFetching = true
            status.isFetched = false
            status.error = false 
            state.predictions = ''
        },
        FETCH_PREDICTIONS_SUCCESS(state, payload) {
            const { status } = state
            status.isFetching = false
            status.isFetched = true
            status.error = false
            state.predictions = payload.payload.res
            state.showPredictedModal = true
        },
        FETCH_PREDICTIONS_ERROR(state) {
            const { status } = state
            status.isFetching = false
            status.isFetched = false
            status.error = true
        },
        SHOW_PREDICTED_MODAL(state) {
            state.showPredictedModal = !state.showPredictedModal
        },
        USER_INPUT_CHANGE(state, payload) {
            state.userInput = payload.payload
        } 
    }
})
export const { FETCH_PREDICTIONS_REQUEST,
               FETCH_PREDICTIONS_SUCCESS,
               FETCH_PREDICTIONS_ERROR,
               SHOW_PREDICTED_MODAL,
               USER_INPUT_CHANGE
             } = mlPredictions.actions

export const fetchMlPredictions = (payload) => async dispatch => {
    dispatch(FETCH_PREDICTIONS_REQUEST())
    try {
        const response = await service.fetchSimilarQuote(payload)
        dispatch(FETCH_PREDICTIONS_SUCCESS(response.data))
    } catch (error) {
        dispatch(FETCH_PREDICTIONS_ERROR())
    }
}

export default mlPredictions.reducer