import { RLAction } from './action'

type Direction = 'up' | 'down' | 'left' | 'right'

class RLController {
  pressed: Direction[] = []
  firePressing: boolean = false

  applyAction(action: RLAction) {
    // 每一步都重置
    this.pressed = []
    this.firePressing = false

    switch (action) {
      case RLAction.UP:
        this.pressed.push('up')
        break
      case RLAction.DOWN:
        this.pressed.push('down')
        break
      case RLAction.LEFT:
        this.pressed.push('left')
        break
      case RLAction.RIGHT:
        this.pressed.push('right')
        break
      case RLAction.FIRE:
        this.firePressing = true
        break
      case RLAction.UP_FIRE:
        this.pressed.push('up')
        this.firePressing = true
        break
      case RLAction.DOWN_FIRE:
        this.pressed.push('down')
        this.firePressing = true
        break
      case RLAction.LEFT_FIRE:
        this.pressed.push('left')
        this.firePressing = true
        break
      case RLAction.RIGHT_FIRE:
        this.pressed.push('right')
        this.firePressing = true
        break
    }
  }
}

export const rlController = new RLController()