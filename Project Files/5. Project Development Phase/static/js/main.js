// Smart Lender - Dashboard Interactive Controller
document.addEventListener("DOMContentLoaded", function () {
    console.log("Smart Lender Dashboard Controller Initialized");

    // --- 1. Theme Switcher System ---
    const toggleSwitch = document.querySelector('#themeToggleCheckbox');
    const settingsThemeToggle = document.querySelector('#settingsThemeToggle');
    const currentTheme = localStorage.getItem('theme') || 'light';

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        
        // Sync checkboxes
        if (toggleSwitch) toggleSwitch.checked = (theme === 'dark');
        if (settingsThemeToggle) settingsThemeToggle.checked = (theme === 'dark');
    }

    // Apply saved theme on load
    applyTheme(currentTheme);

    // Bind event listeners to checkbox toggles
    if (toggleSwitch) {
        toggleSwitch.addEventListener('change', function (e) {
            applyTheme(e.target.checked ? 'dark' : 'light');
        });
    }

    if (settingsThemeToggle) {
        settingsThemeToggle.addEventListener('change', function (e) {
            applyTheme(e.target.checked ? 'dark' : 'light');
        });
    }

    // --- 2. Submitting Loader Animation Overlay ---
    const predictionForm = document.getElementById('predictionForm');
    const loaderOverlay = document.getElementById('loaderOverlay');

    if (predictionForm && loaderOverlay) {
        predictionForm.addEventListener('submit', function (event) {
            // Check form validity before displaying the loader
            if (predictionForm.checkValidity()) {
                loaderOverlay.classList.add('active');
            }
        });
    }

    // --- 3. Dashboard Sidebar Section Toggling (Tabs) ---
    const linkEval = document.getElementById('sidebarLoanEvalLink');
    const linkRisk = document.getElementById('sidebarRiskLink');
    const linkSettings = document.getElementById('sidebarSettingsLink');

    const sectEval = document.getElementById('loanEvaluationSection');
    const sectRisk = document.getElementById('riskAnalyticsSection');
    const sectSettings = document.getElementById('settingsSection');

    function showSection(activeLink, activeSection) {
        // Hide all sections
        if (sectEval) sectEval.style.display = 'none';
        if (sectRisk) sectRisk.style.display = 'none';
        if (sectSettings) sectSettings.style.display = 'none';

        // Show target section
        if (activeSection) activeSection.style.display = 'block';

        // Remove active class from links
        [linkEval, linkRisk, linkSettings].forEach(link => {
            if (link) link.classList.remove('active');
        });

        // Add active class to clicked link
        if (activeLink) activeLink.classList.add('active');
    }

    if (linkEval && sectEval) {
        linkEval.addEventListener('click', function (e) {
            e.preventDefault();
            showSection(linkEval, sectEval);
        });
    }

    if (linkRisk && sectRisk) {
        linkRisk.addEventListener('click', function (e) {
            e.preventDefault();
            showSection(linkRisk, sectRisk);
        });
    }

    if (linkSettings && sectSettings) {
        linkSettings.addEventListener('click', function (e) {
            e.preventDefault();
            showSection(linkSettings, sectSettings);
        });
    }

    // --- 4. Currency Formatter ---
    const currencySelect = document.getElementById('currencySelect');
    if (currencySelect) {
        // Load initial currency format
        const savedCurrency = localStorage.getItem('currency') || '$';
        currencySelect.value = savedCurrency;
        updateCurrencySymbols(savedCurrency);

        currencySelect.addEventListener('change', function (e) {
            const sym = e.target.value;
            localStorage.setItem('currency', sym);
            updateCurrencySymbols(sym);
        });
    }

    function updateCurrencySymbols(symbol) {
        const symbols = document.querySelectorAll('.currency-symbol');
        symbols.forEach(el => {
            el.textContent = symbol;
        });
    }

    // --- 5. Settings Option Functions ---
    // Save Settings button
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    const settingsAlert = document.getElementById('settingsAlert');
    const settingsStrictValidation = document.getElementById('settingsStrictValidation');

    // Load strict validation setting on load
    if (settingsStrictValidation) {
        const strictVal = localStorage.getItem('strictValidation') !== 'false';
        settingsStrictValidation.checked = strictVal;
    }

    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', function () {
            try {
                // Save theme setting
                const themeVal = settingsThemeToggle && settingsThemeToggle.checked ? 'dark' : 'light';
                applyTheme(themeVal);

                // Save currency setting
                if (currencySelect) {
                    const currencyVal = currencySelect.value;
                    localStorage.setItem('currency', currencyVal);
                    updateCurrencySymbols(currencyVal);
                }

                // Save validation setting
                if (settingsStrictValidation) {
                    localStorage.setItem('strictValidation', settingsStrictValidation.checked);
                }

                // Display success banner
                if (settingsAlert) {
                    settingsAlert.classList.remove('d-none');
                    setTimeout(() => {
                        settingsAlert.classList.add('d-none');
                    }, 3000);
                }
            } catch (err) {
                console.error("Failed to save settings:", err);
                alert("An error occurred while saving settings.");
            }
        });
    }

    // Reset Form button
    const resetFormSettingsBtn = document.getElementById('resetFormSettingsBtn');
    if (resetFormSettingsBtn && predictionForm) {
        resetFormSettingsBtn.addEventListener('click', function () {
            predictionForm.reset();
            predictionForm.classList.remove('was-validated');
            // Re-apply currency format to placeholders if reset
            const currentCurrency = localStorage.getItem('currency') || '$';
            updateCurrencySymbols(currentCurrency);
            
            // Temporary confirmation toast/alert in banner
            if (settingsAlert) {
                settingsAlert.classList.remove('alert-success');
                settingsAlert.classList.add('alert-warning');
                settingsAlert.querySelector('span') || (settingsAlert.innerHTML = '<i class="fa-solid fa-triangle-exclamation me-2"></i> Form inputs cleared successfully!');
                settingsAlert.classList.remove('d-none');
                setTimeout(() => {
                    settingsAlert.classList.add('d-none');
                    // Reset back to original HTML
                    settingsAlert.classList.remove('alert-warning');
                    settingsAlert.classList.add('alert-success');
                    settingsAlert.innerHTML = '<i class="fa-solid fa-circle-check me-2"></i> Settings saved successfully!';
                }, 3000);
            }
        });
    }

    // --- 6. Read Tab Parameter from URL ---
    const urlParams = new URLSearchParams(window.location.search);
    const tabParam = urlParams.get('tab');
    if (tabParam === 'risk' && linkRisk && sectRisk) {
        showSection(linkRisk, sectRisk);
    } else if (tabParam === 'settings' && linkSettings && sectSettings) {
        showSection(linkSettings, sectSettings);
    }
});
