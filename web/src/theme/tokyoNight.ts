import { createTheme, Theme } from '@mui/material/styles';

// Tokyo Night color palette
const tokyoNightColors = {
    // Background colors
    background: {
        primary: '#1a1b26',
        secondary: '#16161e',
        elevated: '#24283b',
        surface: '#1f2335',
    },
    // Text colors
    text: {
        primary: '#c0caf5',
        secondary: '#9aa5ce',
        muted: '#565f89',
    },
    // Accent colors
    accent: {
        blue: '#7aa2f7',
        purple: '#bb9af7',
        green: '#9ece6a',
        yellow: '#e0af68',
        red: '#f7768e',
        orange: '#ff9e64',
        cyan: '#7dcfff',
    },
    // UI element colors
    ui: {
        border: '#414868',
        borderLight: '#565f89',
        selection: '#364a82',
        highlight: '#3d59a1',
    }
};

const createTokyoNightTheme = (mode: 'light' | 'dark' = 'dark'): Theme => {
    const isDark = mode === 'dark';

    const lightColors = {
        background: {
            primary: '#f7f7f7',
            secondary: '#ffffff',
            elevated: '#ffffff',
            surface: '#f5f5f5',
        },
        text: {
            primary: '#2c2c2c',
            secondary: '#4c4c4c',
            muted: '#7c7c7c',
        },
        accent: {
            blue: '#0f62fe',
            purple: '#8a3ffc',
            green: '#198038',
            yellow: '#f1c21b',
            red: '#da1e28',
            orange: '#ff832b',
            cyan: '#1192e8',
        },
        ui: {
            border: '#e0e0e0',
            borderLight: '#d0d0d0',
            selection: '#e8f3ff',
            highlight: '#d4edda',
        }
    };

    const colors = isDark ? tokyoNightColors : lightColors;

    return createTheme({
        palette: {
            mode,
            primary: {
                main: colors.accent.blue,
                dark: isDark ? '#5a7fd7' : '#073590',
                light: isDark ? '#9cb8ff' : '#4589ff',
            },
            secondary: {
                main: colors.accent.purple,
                dark: isDark ? '#9b7bdd' : '#6929c4',
                light: isDark ? '#d5c2ff' : '#be95ff',
            },
            background: {
                default: colors.background.primary,
                paper: colors.background.elevated,
            },
            text: {
                primary: colors.text.primary,
                secondary: colors.text.secondary,
            },
            divider: colors.ui.border,
            success: {
                main: colors.accent.green,
            },
            warning: {
                main: colors.accent.yellow,
            },
            error: {
                main: colors.accent.red,
            },
            info: {
                main: colors.accent.cyan,
            },
        },
        typography: {
            fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
            h1: {
                fontWeight: 700,
                fontSize: '2.125rem',
                lineHeight: 1.2,
            },
            h2: {
                fontWeight: 600,
                fontSize: '1.75rem',
                lineHeight: 1.3,
            },
            h3: {
                fontWeight: 600,
                fontSize: '1.5rem',
                lineHeight: 1.3,
            },
            h4: {
                fontWeight: 600,
                fontSize: '1.25rem',
                lineHeight: 1.4,
            },
            h5: {
                fontWeight: 600,
                fontSize: '1.125rem',
                lineHeight: 1.4,
            },
            h6: {
                fontWeight: 600,
                fontSize: '1rem',
                lineHeight: 1.5,
            },
            body1: {
                fontSize: '1rem',
                lineHeight: 1.6,
            },
            body2: {
                fontSize: '0.875rem',
                lineHeight: 1.6,
            },
            button: {
                textTransform: 'none',
                fontWeight: 600,
            },
        },
        shape: {
            borderRadius: 8,
        },
        components: {
            MuiCssBaseline: {
                styleOverrides: {
                    body: {
                        scrollbarWidth: 'thin',
                        scrollbarColor: `${colors.ui.border} ${colors.background.primary}`,
                        '&::-webkit-scrollbar': {
                            width: '8px',
                        },
                        '&::-webkit-scrollbar-track': {
                            background: colors.background.primary,
                        },
                        '&::-webkit-scrollbar-thumb': {
                            background: colors.ui.border,
                            borderRadius: '4px',
                        },
                        '&::-webkit-scrollbar-thumb:hover': {
                            background: colors.ui.borderLight,
                        },
                    },
                },
            },
            MuiPaper: {
                styleOverrides: {
                    root: {
                        backgroundColor: colors.background.elevated,
                        borderColor: colors.ui.border,
                    },
                },
            },
            MuiAppBar: {
                styleOverrides: {
                    root: {
                        backgroundColor: colors.background.elevated,
                        borderBottom: `1px solid ${colors.ui.border}`,
                        boxShadow: 'none',
                    },
                },
            },
            MuiButton: {
                styleOverrides: {
                    root: {
                        borderRadius: 6,
                        textTransform: 'none',
                        fontWeight: 600,
                    },
                    contained: {
                        boxShadow: 'none',
                        '&:hover': {
                            boxShadow: 'none',
                        },
                    },
                },
            },
            MuiTextField: {
                styleOverrides: {
                    root: {
                        '& .MuiOutlinedInput-root': {
                            backgroundColor: colors.background.surface,
                            '& fieldset': {
                                borderColor: colors.ui.border,
                            },
                            '&:hover fieldset': {
                                borderColor: colors.ui.borderLight,
                            },
                            '&.Mui-focused fieldset': {
                                borderColor: colors.accent.blue,
                            },
                        },
                    },
                },
            },
            MuiChip: {
                styleOverrides: {
                    root: {
                        backgroundColor: colors.background.surface,
                        color: colors.text.primary,
                        borderColor: colors.ui.border,
                    },
                },
            },
        },
    });
};

export { createTokyoNightTheme, tokyoNightColors };
export type { Theme };
