import { State } from '../reducers'
import { TankRecord } from '../types/TankRecord'
import { BulletRecord } from '../types/BulletRecord'

function normalize(value: number, max: number) {
  return value / max
}

function directionToNumber(dir: string): number {
  switch (dir) {
    case 'up': return 0
    case 'down': return 1
    case 'left': return 2
    case 'right': return 3
    default: return 0
  }
}

export function extractState(state: State): number[] {
  const result: number[] = []

  // ===== 玩家 =====
  const player = state.tanks.find(t => t.side === 'player' && t.alive)

  if (player) {
    result.push(normalize(player.x, 512))
    result.push(normalize(player.y, 512))
    result.push(directionToNumber(player.direction) / 4)
  } else {
    result.push(0, 0, 0)
  }

  // ===== 最近敌人 =====
  const enemies = state.tanks.filter(t => t.side === 'bot' && t.alive)

  let nearestEnemy: TankRecord | null = null
  let minDist = Infinity

  if (player) {
    enemies.forEach(e => {
      const dx = e.x - player.x
      const dy = e.y - player.y
      const dist = dx * dx + dy * dy

      if (dist < minDist) {
        minDist = dist
        nearestEnemy = e
      }
    })
  }

  if (nearestEnemy) {
    result.push(normalize(nearestEnemy.x, 512))
    result.push(normalize(nearestEnemy.y, 512))
    result.push(directionToNumber(nearestEnemy.direction) / 4)
  } else {
    result.push(0, 0, 0)
  }

  // ===== 最近子弹 =====
  const bullets = state.bullets

  let nearestBullet: BulletRecord | null = null
  minDist = Infinity

  if (player) {
    bullets.forEach(b => {
      const dx = b.x - player.x
      const dy = b.y - player.y
      const dist = dx * dx + dy * dy

      if (dist < minDist) {
        minDist = dist
        nearestBullet = b
      }
    })
  }

  if (nearestBullet) {
    result.push(normalize(nearestBullet.x, 512))
    result.push(normalize(nearestBullet.y, 512))
    result.push(directionToNumber(nearestBullet.direction) / 4)
  } else {
    result.push(0, 0, 0)
  }

  // ===== 剩余敌人 =====
  result.push(enemies.size / 20)

  return result
}