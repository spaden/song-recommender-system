import axios from "axios";

const interceptor = axios.create({
    baseURL: 'http://localhost:8002'
})

export default interceptor 