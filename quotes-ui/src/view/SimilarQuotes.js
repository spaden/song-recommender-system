import './styles/SimilarQuotes.scss'
import { fetchMlPredictions } from '../store/similarPred'
import { useSelector, useDispatch } from 'react-redux'
import Button from '../components/button/button'
import UserInputModal from '../features/userInput/userInputModal'
import LoaderComponent from '../components/loader/loaderComponent'

function SimilarQuotes(props) {
    
    const dispatch = useDispatch()
    const status = useSelector(state => state.mlPrediction.status)
    const userStatus = useSelector(state => state.userStore.status)
    const isAuthenticated = useSelector(state => state.userStore.isAuthenticated)
    const userQuote = useSelector(state => state.mlPrediction.userInput)

    const fetchMlPrediction = () => {
      if (isAuthenticated) {
        dispatch(fetchMlPredictions({
          userInput: userQuote
        }))
      }
    }

    return (
      <div className="similarquotes">
          <div className="container-fluid">
              <div className="row justify-content-center">
                  <div className="col col-sm-12 col-md-10 col-lg-7">
                    <div className={`similarquotes__form
                                     ${userStatus.isFetching || userStatus.error ? 'similarquotes__form--blur' : ''}
                                   `}>
                        <UserInputModal/>
                        <div className="similarquotes__form--button">
                            <Button name="Get a Song!"
                                    onClick={fetchMlPrediction}
                                    isFetching={status.isFetching}></Button>
                        </div>
                    </div>
                    <div className={`similarquotes__loader
                                    ${userStatus.isFetched  || userStatus.error ? 'similarquotes__loader--nodisplay' : ''}
                                   `}>
                        <LoaderComponent/>
                    </div>
                    <div className={ `similarquotes__error
                                      ${userStatus.error ? 'similarquotes__error--display' : ''}
                                    `}>
                        Some error occured, please try again later
                    </div>
                  </div>
              </div>
          </div>
      </div>
    );
  }

  export default SimilarQuotes