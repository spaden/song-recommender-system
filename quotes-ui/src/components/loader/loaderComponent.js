import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSpinner } from '@fortawesome/free-solid-svg-icons'
import './loaderComponent.scss'

function loaderComponent(props) {
    
    return (
      <div className="loadercomponent">
          <div className="loadercomponent__spinner">
            <FontAwesomeIcon icon={faSpinner} />
          </div>
      </div>
    );
  }

  export default loaderComponent