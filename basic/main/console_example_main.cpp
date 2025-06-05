/* Basic console example (esp_console_repl API)

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <stdio.h>
#include <string.h>
#include "esp_system.h"
#include "esp_log.h"
#include "esp_console.h"
#include "esp_vfs_dev.h"
#include "esp_vfs_fat.h"
#include "cmd_system.h"
#include "include/matops.h"
#include "include/linearModel.h"

#if SOC_USB_SERIAL_JTAG_SUPPORTED
#if !CONFIG_ESP_CONSOLE_SECONDARY_NONE
#warning "A secondary serial console is not useful when using the console component. Please disable it in menuconfig."
#endif
#endif

static const char *TAG = "example";
#define PROMPT_STR CONFIG_IDF_TARGET

long double delta_t = 0.1;
long double meas_rul = 0.0;
// Linear model for predicting transformer RUL
// state vector: [rul]
// transition matrix: [1]
// input matrix: [-1]
// input vector: [delta_t]
// measurement matrix: [1]
// measurement vector: [rul]

LinearModel lm = LinearModel(
    // motion starts at rest
    new matrix{1, 1, new long double[1]{180000.0}},
    new matrix{1, 1, new long double[1]{1.0}},
    new matrix{1, 1, new long double[1]{-1.0}});

// initialize
// state covariance matrix
matrix *stateCOVMat = new matrix{1, 1, new long double[1]{0.01}};
// process noise covariance matrix
matrix *processCOVMat = new matrix{1, 1, new long double[1]{10.0}};
// measurement noise covariance matrix
matrix *measurementCOVMat = new matrix{1, 1, new long double[1]{0.01}};
// kalman gain matrix
matrix *kalmanGainMat = new matrix{1, 1, new long double[1]{0.01}};
// measurement matrix
matrix *measurementMat = new matrix{1, 1, new long double[1]{1.0}};

static int float_input_handler(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: float_input <value>\n");
        return 1;
    }
    float inp_value = atof(argv[1]);
    // float value = strtof(argv[1], NULL);
    printf("Received RUL value: %f\n", inp_value);

    printf("Running single step of KF......\n");

    meas_rul = inp_value; // Take the float input as measurement RUL
    // For testing, we can use a fixed delta_t

    // Take the float input as measurement RUL and estimate the next state

    // ****PART 1: PREDICTION STEP
    matrix *input_vec = new matrix{1, 1, new long double[1]{delta_t}};
    // Predict the next state vector: x = Fx + Bu
    lm.updateState(input_vec);
    // z_pred = Hx
    matrix *predMeasurement = mat_mul(measurementMat, lm.getStateVector());

    // P = FPF^T + Q, where Q is process noise
    stateCOVMat = mat_mul(lm.getTransitionMatrix(), mat_mul(stateCOVMat, mat_transpose(lm.getTransitionMatrix())));
    stateCOVMat = mat_add(stateCOVMat, processCOVMat);

    // ****PART 2: UPDATE STEP
    // Calculate the Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
    matrix *temp = mat_mul(mat_mul(measurementMat, stateCOVMat), mat_transpose(measurementMat));
    temp = mat_add(temp, measurementCOVMat);
    kalmanGainMat = mat_mul(mat_mul(stateCOVMat, mat_transpose(measurementMat)), mat_inv(temp));

    // innovation/error: y_err = z - Hx, where measurement vector z is supplied externally.
    matrix *y_err = mat_sub(new matrix{1, 1, new long double[1]{meas_rul}}, predMeasurement);

    // correction step: x = x + K*y_err
    lm.setStateVector(mat_add(lm.getStateVector(), mat_mul(kalmanGainMat, y_err)));

    // P = (I - KH)P
    stateCOVMat = mat_mul(
        mat_sub(new matrix{1, 1, new long double[1]{1.0}},
                mat_mul(kalmanGainMat, measurementMat)),
        stateCOVMat);
    printf("State vector after correction: %Lf\n", lm.getStateVector()->data[0]);

    // free the matrices
    free(input_vec->data);

    return 0;
}

static esp_console_cmd_t float_input_cmd = {
    .command = "float_input",
    .help = "Input a float value",
    .hint = NULL,
    .func = &float_input_handler,
    .argtable = NULL};
void register_float_input_command(void)
{
    esp_err_t err = esp_console_cmd_register(&float_input_cmd);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to register float_input command: %s", esp_err_to_name(err));
    }
    else
    {
        ESP_LOGI(TAG, "Registered float_input command");
    }
}

extern "C" void app_main(void)
{
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    /* Prompt to be printed before each line.
     * This can be customized, made dynamic, etc.
     */
    repl_config.prompt = PROMPT_STR ">";
    repl_config.max_cmdline_length = CONFIG_CONSOLE_MAX_COMMAND_LINE_LENGTH;

    /* Register commands */
    esp_console_register_help_command();
    register_system_common();
    register_system_sleep();
    register_float_input_command();

#if defined(CONFIG_ESP_CONSOLE_UART_DEFAULT) || defined(CONFIG_ESP_CONSOLE_UART_CUSTOM)
    esp_console_dev_uart_config_t hw_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&hw_config, &repl_config, &repl));

#elif defined(CONFIG_ESP_CONSOLE_USB_CDC)
    esp_console_dev_usb_cdc_config_t hw_config = ESP_CONSOLE_DEV_CDC_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_cdc(&hw_config, &repl_config, &repl));

#elif defined(CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG)
    esp_console_dev_usb_serial_jtag_config_t hw_config = ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&hw_config, &repl_config, &repl));

#else
#error Unsupported console type
#endif

    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
