import 'normalize.css'
import React from 'react'
import ReactDOM from 'react-dom'
import { Provider } from 'react-redux'
import App from './App'
import './battle-city.css'
import store from './utils/store'
import { TankEnv } from './rl/env'
import { RLAction } from './rl/action'

ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById('container'),
)

if (process.env.NODE_ENV === 'development') {
  const env = new TankEnv()

  setTimeout(() => {
    env.reset()

    let stepCount = 0

    const loop = () => {
      const action = Math.floor(Math.random() * 10)

      const result = env.step(action)

      stepCount++

      if (result.done) {
        console.log('Episode finished at step', stepCount)
        return
      }

      requestAnimationFrame(loop)
    }

    loop()
  }, 1000)
}