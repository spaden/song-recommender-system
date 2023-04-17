import './inputComponentStyle.scss'

function InputComponent(props) {
    
    const onInputChange = (e) => {
      // props.onChange(e.currentTarget.textContent)
    }
    
    let checkStatus = () => {
      return props.isFetching || props.isModalShown
    }

    return (
      <div className="inputcomponent">
        <div className={`inputcomponent__textarea 
                        ${checkStatus() ? 'inputcomponent__textarea--fetching': ''}
                      `}
             contentEditable={!checkStatus()}
             onInput={props.onInputChange}
             onClick={props.onClick}
             onBlur={props.onBlur}>
               {props.placeholder}
        </div>
      </div>
    );
  }

  export default InputComponent