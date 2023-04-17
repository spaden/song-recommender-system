import interceptor from '../utilities/interceptor'

const initRequest = (payload) => {
    return interceptor.post('/loginUser', payload)
}

const fetchSimilarQuote = (payload) => {
    return interceptor.post('/getSimilarQuote', payload)
}

const service = {
    initRequest,
    fetchSimilarQuote
}

export default service 