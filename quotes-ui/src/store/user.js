import { createSlice } from '@reduxjs/toolkit'
import service from '../service/service'

const initialState = {
    userName: 'Max',
    password: 'strange',
    isAuthenticated: false,
    status: {
        isFetching: false,
        isFetched: false,
        error: false
    }
}

const userStore = createSlice({
    name: 'userStore',
    initialState: initialState,
    reducers: {
        FETCH_USER_REQUEST(state) {
            const { status } = state
            status.isFetching = true
            status.isFetched = false
            status.error = false 
            state.isAuthenticated = false
        },
        FETCH_USER_SUCCESS(state, responseData) {
            const { status } = state
            status.isFetching = false
            status.isFetched = true
            status.error = false
            const { payload } = responseData
            state.isAuthenticated = payload
            console.log(state.isAuthenticated)
        },
        FETCH_USER_ERROR(state) {
            const { status } = state
            status.isFetching = false
            status.isFetched = false
            status.error = true
        }
    }
})

export const { FETCH_USER_REQUEST,
               FETCH_USER_SUCCESS,
               FETCH_USER_ERROR
             } = userStore.actions

export const fetchInitRequest = (payload) => async dispatch => {
    dispatch(FETCH_USER_REQUEST())
    try {
        const response = await service.initRequest(payload)
        dispatch(FETCH_USER_SUCCESS(response.data))
    } catch (error) {
        dispatch(FETCH_USER_ERROR())
    }
}

export default userStore.reducer