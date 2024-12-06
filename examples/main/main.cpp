#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http  = beast::http;           // from <boost/beast/http.hpp>
namespace asio  = boost::asio;           // from <boost/asio.hpp>
using tcp       = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#    include <signal.h>
#    include <unistd.h>
#elif defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <signal.h>
#    include <windows.h>
#endif

#if defined(_MSC_VER)
#    pragma warning(disable : 4244 4267)  // possible loss of data
#endif

static llama_context **           g_ctx;
static llama_model **             g_model;
static common_sampler **          g_smpl;
static common_params *            g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream *       g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool                       is_interacting  = false;
static bool                       need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    // (void) argc;

    // LOG("\nexample usage:\n");
    // LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    // LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    // LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            // LOG("\n");
            // common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

static std::string chat_add_and_format(struct llama_model * model, std::vector<common_chat_msg> & chat_msgs,
                                       const std::string & role, const std::string & content) {
    common_chat_msg new_msg{ role, content };
    auto formatted = common_chat_format_single(model, g_params->chat_template, chat_msgs, new_msg, role == "user");
    chat_msgs.push_back({ role, content });
    LOG_DBG("formatted: '%s'\n", formatted.c_str());
    return formatted;
}

std::string run_model(bool add_bos, int n_ctx, std::string path_session, common_params params, llama_model * model,
                      std::vector<llama_token> session_tokens, llama_context * ctx, struct common_sampler * smpl,
                      const std::string & prompt);

std::string run_model(bool add_bos, int n_ctx, std::string path_session, common_params params, llama_model * model,
                      std::vector<llama_token> session_tokens, llama_context * ctx, struct common_sampler * smpl,
                      const std::string & prompt) {
    std::vector<llama_token>     embd_inp;
    std::vector<common_chat_msg> chat_msgs;

    {
        if (params.interactive_first || !params.prompt.empty() || session_tokens.empty()) {
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            embd_inp = session_tokens;
        }
    }

    // // Should not run without any tokens
    // if (embd_inp.empty()) {
    //     if (add_bos) {
    //         embd_inp.push_back(llama_token_bos(model));
    //         LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
    //     } else {
    //         LOG_ERR("input is empty\n");
    //         return "";
    //     }
    // }
    //
    // // Tokenize negative prompt
    // if ((int) embd_inp.size() > n_ctx - 4) {
    //     LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
    //     return "";
    // }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int) embd_inp.size();
    } else {
        params.n_keep += add_bos;  // always keep the BOS token
    }
    // ctrl+C handling
    {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset(&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return "";
    }

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0 && "grp_attn_n must be positive");                          // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n");  // NOLINT
        //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
    }

    std::ostringstream response;
    bool               is_antiprompt        = false;
    bool               input_echo           = true;
    bool               display              = true;
    bool               need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int> input_tokens;
    g_input_tokens = &input_tokens;
    std::vector<int> output_tokens;
    g_output_tokens = &output_tokens;
    std::ostringstream output_ss;
    g_output_ss = &output_ss;
    std::ostringstream assistant_ss;  // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;
    antiprompt_ids.reserve(params.antiprompt.size());
    if (llama_model_has_encoder(model)) {
        int           enc_input_size = embd_inp.size();
        llama_token * enc_input_buf  = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return "";
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = llama_token_bos(model);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                embd.resize(max_embd_size);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift) {
                        LOG_DBG("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_DBG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left / 2;

                    LOG_DBG(
                        "context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_kv_cache_seq_rm(ctx, 0, params.n_keep, params.n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n * ga_i) / ga_w;
                    const int bd = (ga_w / ga_n) * (ga_n - 1);
                    const int dd = (ga_w / ga_n) - ib * bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib * bd, ga_i + ib * bd,
                            n_past + ib * bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n,
                            (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, n_past + ib * bd, dd,
                            ga_i + ib * bd + ga_w + dd, n_past + ib * bd + dd);

                    llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib * bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib * bd + ga_w, n_past + ib * bd, dd);

                    n_past -= bd;

                    ga_i += ga_w / ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for (; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return "";
                }

                n_past += n_eval;

                LOG_DBG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }
        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                // LOG("%s", token_str.c_str());
                response << token_str.c_str();

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int         n_prev      = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos =
                        last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding) ?
                            last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding) :
                            0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                llama_token last_token = common_sampler_last(smpl);
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (llama_token_is_eog(model, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");
            }

            // if current token is not EOG, we add it to current assistant message
            if (params.conversation) {
                const auto id = common_sampler_last(smpl);
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            if (n_past > 0 && is_interacting) {
                LOG_DBG("waiting for user input\n");

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(model));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    // LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                display = params.display_prompt;

                std::string line;

                // done taking input, reset color
                display = true;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        // LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool        format_chat = params.conversation && params.enable_chat_template;
                    std::string user_inp =
                        format_chat ? chat_add_and_format(model, chat_msgs, "user", buffer) : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_token_eot(model);
                        embd_inp.push_back(eot == -1 ? llama_token_eos(model) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << common_token_to_piece(ctx, token);
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                } else {
                    LOG_DBG("empty line, passing control back\n");
                }

                input_echo = false;  // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !(params.interactive)) {
            // LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain       = params.n_predict;
            is_interacting = true;
        }
    }

    // common_sampler_free(smpl);

    return response.str();

    // return 0;
}

struct common_init_result common_init_from_params_cached_model(common_params & params, llama_model * model);

struct common_init_result common_init_from_params_cached_model(common_params & params, llama_model * model) {
    common_init_result iparams;

    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.c_str());
        return iparams;
    }

    if (params.reranking) {
        bool ok = true;

        if (llama_token_bos(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have a  BOS token, reranking will not work\n", __func__);
            ok = false;
        }

        if (llama_token_eos(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have an EOS token, reranking will not work\n", __func__);
            ok = false;
        }

        if (llama_token_sep(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have a  SEP token, reranking will not work\n", __func__);
            ok = false;
        }

        if (!ok) {
            llama_free_model(model);

            return iparams;
        }
    }

    auto cparams = common_context_params_to_llama(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == NULL) {
        LOG_ERR("%s: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return iparams;
    }

    if (params.ctx_shift && !llama_kv_cache_can_shift(lctx)) {
        LOG_ERR("%s: KV cache shifting is not supported for this model (--no-context-shift to disable)'\n", __func__);
        llama_free_model(model);
        return iparams;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) {
            params.control_vector_layer_start = 1;
        }
        if (params.control_vector_layer_end <= 0) {
            params.control_vector_layer_end = llama_n_layer(model);
        }

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            llama_free_model(model);

            return iparams;
        }

        int err = llama_control_vector_apply(lctx, cvec.data.data(), cvec.data.size(), cvec.n_embd,
                                             params.control_vector_layer_start, params.control_vector_layer_end);
        if (err) {
            llama_free(lctx);
            llama_free_model(model);

            return iparams;
        }
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        common_lora_adapter_container loaded_la;
        loaded_la.path    = la.path;
        loaded_la.scale   = la.scale;
        loaded_la.adapter = llama_lora_adapter_init(model, la.path.c_str());
        if (loaded_la.adapter == nullptr) {
            LOG_ERR("%s: failed to apply lora adapter '%s'\n", __func__, la.path.c_str());
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }
        iparams.lora_adapters.push_back(loaded_la);  // copy to list of loaded adapters
    }
    if (!params.lora_init_without_apply) {
        common_lora_adapters_apply(lctx, iparams.lora_adapters);
    }

    if (params.sampling.ignore_eos && llama_token_eos(model) == LLAMA_TOKEN_NULL) {
        LOG_WRN("%s: warning: model does not have an EOS token, ignoring --ignore-eos\n", __func__);
        params.sampling.ignore_eos = false;
    }

    if (params.warmup) {
        LOG_WRN("%s: warming up the model with an empty run - please wait ... (--no-warmup to disable)\n", __func__);

        std::vector<llama_token> tmp;
        llama_token              bos = llama_token_bos(model);
        llama_token              eos = llama_token_eos(model);
        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == -1) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
        }
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);
    }

    iparams.model   = model;
    iparams.context = lctx;

    return iparams;
}

int main(int argc, char ** argv) {
    common_params params;

    llama_model *    model = nullptr;
    llama_context *  ctx   = nullptr;
    common_sampler * smpl  = nullptr;

    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;
    set_process_priority(params.cpuparams.priority);
    llama_numa_init(params.numa);

    llama_backend_init();

    common_init_result           llama_init = common_init_from_params(params);
    std::vector<common_chat_msg> chat_msgs;
    model   = llama_init.model;
    ctx     = llama_init.context;
    g_model = &model;
    g_ctx   = &ctx;
    g_smpl  = &smpl;
    smpl    = common_sampler_init(model, sparams);

    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_add_eos_token(model));
    }
    const bool add_bos = llama_add_bos_token(model);

    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    auto * ggml_threadpool_new_fn =
        (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn =
        (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch        = ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp              = ggml_threadpool_params_from_cpu_params(params.cpuparams);
    struct ggml_threadpool *      threadpool_batch = NULL;

    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
    std::string              path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    try {
        asio::io_context ioc;
        tcp::acceptor    acceptor(ioc, tcp::endpoint(tcp::v4(), 8080));
        std::cout << "Server is running on http://localhost:8080" << std::endl;
        while (true) {
            common_init_result llama_init = common_init_from_params_cached_model(params, model);
            model                         = llama_init.model;
            ctx                           = llama_init.context;
            const int n_ctx               = llama_n_ctx(ctx);

            tcp::socket socket(ioc);
            acceptor.accept(socket);

            beast::flat_buffer               buffer;
            http::request<http::string_body> req;
            http::read(socket, buffer, req);

            http::response<http::string_body> res;

            if (req.method() == http::verb::post) {
                std::string input = req.body();
                std::string response_output =
                    run_model(add_bos, n_ctx, path_session, params, model, session_tokens, ctx, smpl, input);

                res.result(http::status::ok);
                res.set(http::field::content_type, "text/plain");
                res.body() = response_output;
                res.prepare_payload();
            } else {
                res.result(http::status::method_not_allowed);
                res.set(http::field::content_type, "text/plain");
                res.body() = "Only POST method is supported.";
                res.prepare_payload();
            }

            http::write(socket, res);

            // Close the socket
            beast::error_code ec;
            socket.shutdown(tcp::socket::shutdown_send, ec);
        }
    } catch (std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();
    return 0;
}
