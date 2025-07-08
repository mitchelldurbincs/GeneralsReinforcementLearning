package input

import "github.com/hajimehoshi/ebiten/v2"

func IsLeftClickJustPressed() bool {
	return ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)
}

func IsRightClickJustPressed() bool {
	return ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight)
}

func GetCursorPosition() (int, int) {
	return ebiten.CursorPosition()
}