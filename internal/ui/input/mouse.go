package input

import (
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

func IsLeftClickJustPressed() bool {
	return inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft)
}

func IsRightClickJustPressed() bool {
	return inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight)
}

func GetCursorPosition() (int, int) {
	return ebiten.CursorPosition()
}