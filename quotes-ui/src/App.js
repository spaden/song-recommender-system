import { Route } from 'react-router-dom'
import './App.css'
import Home from './view/Home'
import SimilarQuotes from './view/SimilarQuotes'
import { fetchInitRequest } from './store/user'
import { useSelector, useDispatch } from 'react-redux'
import { useEffect } from 'react'

function App() {
  
  const dispatch = useDispatch()
  const userState = useSelector(state => state.userStore)

  useEffect(() => {
    dispatch(fetchInitRequest({
      name: userState.userName,
      password: userState.password
    }))
  }, [])

  return (
    <div className="container-fluid justify-content-center">
        <Route path="/" exact>
          <SimilarQuotes/>
        </Route>
    </div>
  );
}

export default App;
