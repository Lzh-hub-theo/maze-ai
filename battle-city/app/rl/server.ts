import express from 'express'
import { TankEnv } from './env'

const app = express()
app.use(express.json())

const env = new TankEnv()

// reset
app.post('/reset', (req: Request, res: Response) => {
  const state = env.reset()
  res.json(state)
})

// step
app.post('/step', (req: Request, res: Response) => {
  const { action } = req.body
  const result = env.step(action)

  res.json(result)
})

app.listen(3000, () => {
  console.log('RL server running on port 3000')
})