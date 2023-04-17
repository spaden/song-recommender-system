import './userInputModal.scss'
import LoaderComponent from '../../components/loader/loaderComponent'
import { SHOW_PREDICTED_MODAL, USER_INPUT_CHANGE } from '../../store/similarPred'
import { useSelector, useDispatch } from 'react-redux'
import InputComponent from '../../components/inputcomponent/inputComponent'

function UserInputModal(props) {
    const dispatch = useDispatch()
    
    const showPredictedModal = useSelector(state => state.mlPrediction.showPredictedModal)
    const status = useSelector(state => state.mlPrediction.status)
    const predictions = useSelector(state => state.mlPrediction.predictions)
    
    const placeholder = 'Eloborate your feelings and I try to reciprocate with a song!'

    const checkIfEmpty = (e) => {
        const textConent = e.currentTarget.textContent
        if (textConent === placeholder || textConent === '') {
          e.currentTarget.textContent = placeholder
        }
    }
  
    const checkIfPlaceholder = (e) => {
        const textContent = e.currentTarget.textContent
        if (textContent === placeholder) {
          e.currentTarget.textContent = ''
        }
    }

    const inputChange = (e) => {
      const textContent = e.currentTarget.textContent
      dispatch(USER_INPUT_CHANGE(textContent))
    }

    const showModal = () => {
        dispatch(SHOW_PREDICTED_MODAL())
    }
    const showLoader = () => {
        
          return (
            <div className="userinputmodal__spinnercomponent">
              <LoaderComponent/>
            </div>
          )
        
    }

    const predictedQuote = () => {
        return (
          <div className="userinputmodal__predictedQuote"
               contentEditable="false">
              <span className="userinputmodal__predictedQuote--close"
                    onClick={showModal}>
                X
              </span>
              <div className="userinputmodal__predictedQuote--quotetext">
                  <a href={predictions} target="_blank">{predictions}</a>
              </div>
          </div>
        )
    }

    const renderComponent = () => {
        if (status.isFetching) {
            return showLoader()
        } else if (status.isFetched && predictions) {
            if (showPredictedModal) {
                return predictedQuote()
            } else {
                return ''
            }
        }else {
            return ''
        }
    }
    

    return (
      <div className="userinputmodal">
          <InputComponent onClick={checkIfPlaceholder}
                          onBlur={checkIfEmpty}
                          onInputChange={inputChange}
                          placeholder={placeholder}
                          isFetching={status.isFetching}
                          isModalShown={showPredictedModal}/>
         {renderComponent()}
      </div>
    );
  }

  export default UserInputModal
  