import store from '../utils/store'
import * as actions from '../utils/actions'
import { State } from '../reducers'
import { rlController } from './controller'
import { RLAction } from './action'
import { extractState } from './state'

const FIXED_DELTA = 16 // ~60FPS

export class TankEnv {
  private prevState: State | null = null

  reset(): State {
    store.dispatch({ type: actions.A.ResetGame })
    store.dispatch({ type: actions.A.StartGame })
    const state = store.getState()
    this.prevState = state
    return extractState(state)
  }

  step(action: RLAction) {
    // 1️⃣ 应用AI动作（写在 controller.ts）
    rlController.applyAction(action)

    // 2️⃣ 推进一帧
    store.dispatch(actions.tick(FIXED_DELTA))
    store.dispatch(actions.afterTick(FIXED_DELTA))

    // 3️⃣ 获取状态
    const state = store.getState()

    let reward = 0
    if (this.prevState) {
      reward = this.computeReward(this.prevState, state)
    }
    this.prevState = state

    return {
      state: extractState(state), // ✅ 核心
      reward: 0,
      done: this.isDone(state)
    }
  }

  private computeReward(prev: State, curr: State): number {
    let reward = 0

    const prevPlayer = prev.tanks.find(t => t.side === 'player')
    const currPlayer = curr.tanks.find(t => t.side === 'player')

    // ===== 玩家死亡 =====
    if (prevPlayer && currPlayer && prevPlayer.alive && !currPlayer.alive) {
      reward -= 10
    }

    // ===== 被击中（hp减少）=====
    if (prevPlayer && currPlayer && currPlayer.hp < prevPlayer.hp) {
      reward -= 1
    }

    // ===== 击杀敌人 =====
    const prevEnemies = prev.tanks.filter(t => t.side === 'bot' && t.alive)
    const currEnemies = curr.tanks.filter(t => t.side === 'bot' && t.alive)

    if (currEnemies.size < prevEnemies.size) {
      reward += 5 * (prevEnemies.size - currEnemies.size)
    }

    // ===== 命中敌人（hp下降）=====
    prevEnemies.forEach(prevE => {
      const currE = curr.tanks.get(prevE.tankId)
      if (currE && currE.hp < prevE.hp) {
        reward += 1
      }
    })

    // ===== 存活奖励 =====
    reward += 0.01

    // ===== 时间惩罚 =====
    reward -= 0.01

    return reward
  }

  private isDone(state: State): boolean {
    return state.game.status === 'gameover'
  }
}
