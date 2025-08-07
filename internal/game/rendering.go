package game

import (
	"strings"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

// This file contains all board rendering functionality for the game engine.

// ANSI color codes (Unchanged - for Board rendering)
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorWhite  = "\033[37m"
	ColorGray   = "\033[90m"

	BgRed    = "\033[41m"
	BgGreen  = "\033[42m"
	BgYellow = "\033[43m"
	BgBlue   = "\033[44m"
	BgPurple = "\033[45m"
	BgCyan   = "\033[46m"
)

var playerColors = []string{ColorRed, ColorBlue, ColorGreen, ColorYellow, ColorPurple, ColorCyan}

// Board returns a string representation of the board
func (e *Engine) Board(playerID int) string {
	const (
		EmptySymbol    = "·"
		CitySymbol     = "⬢"
		GeneralSymbol  = "♔"
		MountainSymbol = "▲"
	)

	// Pre-allocate buffer size based on board dimensions
	// Each cell takes roughly: 2 chars for symbol + ~20 chars for ANSI codes
	// Plus headers and legend
	width := e.gs.Board.W
	height := e.gs.Board.H
	estimatedSize := (width*22+10)*(height+3) + 100 // Extra for headers and legend

	var sb strings.Builder
	sb.Grow(estimatedSize)

	// Header row
	sb.WriteString("    ")
	for x := 0; x < width; x++ {
		sb.WriteString(core.IntToStringFixedWidth(x, 2))
	}
	sb.WriteString("\n")

	// Board rows
	for y := 0; y < height; y++ {
		sb.WriteString(core.IntToStringFixedWidth(y, 2))
		sb.WriteString(" ")
		for x := 0; x < width; x++ {
			t := e.gs.Board.T[e.gs.Board.Idx(x, y)]
			e.getTileDisplayDirect(&sb, t, playerID)
		}
		sb.WriteString("\n")
	}

	// Legend
	sb.WriteString("\n")
	sb.WriteString(EmptySymbol)
	sb.WriteString("=empty ")
	sb.WriteString(CitySymbol)
	sb.WriteString("=city ")
	sb.WriteString(GeneralSymbol)
	sb.WriteString("=general ")
	sb.WriteString(MountainSymbol)
	sb.WriteString("=mountain A-H=players\n")

	return sb.String()
}

// getTileDisplayDirect writes the tile display directly to the strings.Builder to avoid allocations
func (e *Engine) getTileDisplayDirect(sb *strings.Builder, t core.Tile, playerID int) {
	const (
		EmptySymbol    = "·"
		CitySymbol     = "⬢"
		GeneralSymbol  = "♔"
		MountainSymbol = "▲"
		PlayerSymbols  = "ABCDEFGH"
	)

	visible := playerID < 0 || !e.gs.FogOfWarEnabled || t.IsVisibleTo(playerID)

	// Write color first
	if !visible {
		sb.WriteString(ColorGray)
		sb.WriteString(" ")
	} else if t.IsMountain() {
		sb.WriteString(ColorGray)
		sb.WriteString(" ")
		sb.WriteString(MountainSymbol)
	} else if t.IsGeneral() {
		sb.WriteString(getPlayerColor(t.Owner))
		sb.WriteByte(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		sb.WriteString(GeneralSymbol)
	} else if t.IsCity() && t.IsNeutral() {
		sb.WriteString(ColorWhite)
		sb.WriteString(" ")
		sb.WriteString(CitySymbol)
	} else if t.IsCity() {
		sb.WriteString(getPlayerColor(t.Owner))
		sb.WriteByte(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		sb.WriteString(CitySymbol)
	} else if t.IsNeutral() && t.Type == core.TileNormal {
		sb.WriteString(ColorGray)
		if t.Army == 0 {
			sb.WriteString(" ")
			sb.WriteString(EmptySymbol)
		} else if t.Army >= 100 {
			sb.WriteString("++")
		} else if t.Army >= 10 {
			sb.WriteString(core.IntToStringFixedWidth(t.Army, 2))
		} else {
			sb.WriteString(" ")
			sb.WriteString(core.IntToStringFixedWidth(t.Army, 1))
		}
	} else if t.Type == core.TileNormal {
		sb.WriteString(getPlayerColor(t.Owner))
		sb.WriteByte(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		if t.Army >= 100 {
			sb.WriteString("+")
		} else if t.Army >= 10 {
			sb.WriteString(core.IntToStringFixedWidth(t.Army, 1))
		} else {
			sb.WriteString(" ")
		}
	}

	sb.WriteString(ColorReset)
	sb.WriteString(" ")
}

// getTileDisplay returns a colored tile representation - used for consistency with existing code
func (e *Engine) getTileDisplay(t core.Tile, playerID int) (string, string) {
	const (
		EmptySymbol    = "·"
		CitySymbol     = "⬢"
		GeneralSymbol  = "♔"
		MountainSymbol = "▲"
		PlayerSymbols  = "ABCDEFGH"
	)

	visible := playerID < 0 || !e.gs.FogOfWarEnabled || t.IsVisibleTo(playerID)

	if !visible {
		return ColorGray, " "
	}

	if t.IsMountain() {
		return ColorGray, " " + MountainSymbol
	}

	if t.IsGeneral() {
		c := getPlayerColor(t.Owner)
		symbol := string(PlayerSymbols[t.Owner%len(PlayerSymbols)]) + GeneralSymbol
		return c, symbol
	}

	if t.IsCity() && t.IsNeutral() {
		return ColorWhite, " " + CitySymbol
	}

	if t.IsCity() {
		c := getPlayerColor(t.Owner)
		symbol := string(PlayerSymbols[t.Owner%len(PlayerSymbols)]) + CitySymbol
		return c, symbol
	}

	if t.IsNeutral() && t.Type == core.TileNormal {
		if t.Army == 0 {
			return ColorGray, " " + EmptySymbol
		}
		if t.Army >= 100 {
			return ColorGray, "++"
		}
		if t.Army >= 10 {
			return ColorGray, core.IntToStringFixedWidth(t.Army, 2)
		}
		return ColorGray, " " + core.IntToStringFixedWidth(t.Army, 1)
	}

	if t.Type == core.TileNormal {
		c := getPlayerColor(t.Owner)
		symbol := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		if t.Army >= 100 {
			symbol += "+"
		} else if t.Army >= 10 {
			symbol += core.IntToStringFixedWidth(t.Army, 1)
		} else {
			symbol += " "
		}
		return c, symbol
	}

	return "", "??"
}

// getPlayerColor returns the color for the given player ID
func getPlayerColor(playerID int) string {
	if playerID < 0 || playerID >= len(playerColors) {
		return ColorWhite
	}
	return playerColors[playerID]
}
