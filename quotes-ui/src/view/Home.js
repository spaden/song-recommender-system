import { useEffect } from 'react'
import './styles/Home.scss'
import { useSelector, useDispatch } from 'react-redux'
import { FETCH_PREDICTIONS_REQUEST, FETCH_PREDICTIONS_ERROR } from '../store/similarPred'
import InputComponent from '../components/inputcomponent/inputComponent'

function Home(props) {
    const test = useSelector(state => state.mlPrediction.status.isFetching)
    const dispatch = useDispatch()
    const check = () => {
      dispatch(FETCH_PREDICTIONS_REQUEST())
      setTimeout(() => {
        dispatch(FETCH_PREDICTIONS_ERROR())
      }, 5000)
      // 
    }
    useEffect(() => {
      // console.log(test)
    }, [test])
    
    
    
    return (
      <div className="homecontainer">
        Test

      </div>
    );
  }

  export default Home